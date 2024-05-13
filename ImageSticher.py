import glob
import numpy as np
import cv2
import threading
from queue import Queue
from math import sqrt


DATASET = "data"
ROWS = 8
COLUMNS = 9

SCALE_FACTOR = 4

GOOD_THRESH_H = 0.55
GOOD_THRESH_V = 0.42

MASK_PROPORTION_H = 0.2
MASK_PROPORTION_V = 0.3

H_CROP_AMOUNT = 0.2
V_CROP_AMOUNT = 0.2


class image():
    def __init__(self, _image, _path=None):
        self.path = _path
        self.image = _image
        self.h, self.w = self.image.shape[:2]
        self.image_s = cv2.resize(
            self.image, (int(self.w/SCALE_FACTOR), int(self.h/SCALE_FACTOR)))
        self.hs, self.ws = self.image_s.shape[:2]

    @classmethod
    def from_path(cls, _path):
        return cls(cv2.imread(_path), _path)


def import_worker(path, queue):
    queue.put(image.from_path(path))


def stich_worker(row, queue, id):
    result = row[0]
    for i in range(1, len(row)):
        result = stich(result, row[i], "horizontal")
    queue.put((result, id))


def stich(image_1: image, image_2: image, orientation):
    mask1 = np.zeros((image_1.hs, image_1.ws), np.uint8)
    mask2 = np.zeros((image_2.hs, image_2.ws), np.uint8)

    if orientation == "horizontal":
        cv2.rectangle(mask1, (image_1.ws, image_1.hs), (int(image_1.ws-image_2.ws+image_2.ws*MASK_PROPORTION_H), 0), 255, -1)
        cv2.rectangle(mask2, (image_2.ws - int(image_2.ws*MASK_PROPORTION_H), image_2.hs), (0, 0), 255, -1)

    elif orientation == "vertical":
        cv2.rectangle(mask1, (0, int(image_1.hs-image_2.hs+image_2.hs*MASK_PROPORTION_V)), (image_1.ws, image_1.hs), 255, -1)
        cv2.rectangle(mask2, (image_2.ws, image_2.hs - int(image_2.hs*MASK_PROPORTION_V)), (0, 0), 255, -1)

    image1sm = cv2.bitwise_and(image_1.image_s, image_1.image_s, mask=mask1)
    image2sm = cv2.bitwise_and(image_2.image_s, image_2.image_s, mask=mask2)

    sift = cv2.SIFT_create()
    keypoints_image1s = sift.detect(image1sm, None)
    keypoints_image2s = sift.detect(image2sm, None)

    keypoints_image1s, descriptors_image1s = sift.compute(image1sm, keypoints_image1s)
    keypoints_image2s, descriptors_image2s = sift.compute(image2sm, keypoints_image2s)

    brute_force_matcher = cv2.BFMatcher()

    matches = brute_force_matcher.knnMatch(descriptors_image1s, descriptors_image2s, k=2)

    good = []

    if orientation == "horizontal":
        for m, n in matches:
            if m.distance < GOOD_THRESH_H * n.distance:
                good.append(m)

    elif orientation == "vertical":
        for m, n in matches:
            if m.distance < GOOD_THRESH_V * n.distance:
                good.append(m)

    good1 = np.array([keypoints_image1s[m.queryIdx].pt for m in good])
    good2 = np.array([keypoints_image2s[m.trainIdx].pt for m in good])
 
    M, _ = cv2.findHomography(good2, good1, cv2.RANSAC, 5.0)

    d1 = sqrt(M[0, 0]**2 + M[0, 1]**2)
    d2 = sqrt(M[1, 0]**2 + M[1, 1]**2)

    M[0, 0] = M[0, 0]/d1; M[0, 1] = 0;          M[0, 2] *= SCALE_FACTOR
    M[1, 0] = 0;          M[1, 1] = M[1, 1]/d2; M[1, 2] *= SCALE_FACTOR
    M[2, 0] = 0;          M[2, 1] = 0;          M[2, 2] = 1

    print(M)

    if M[0, 2] < 0:
        x = image_1.w
    else:
        x = int(image_2.w + abs(M[0, 2]))

    if x < max(image_1.w, image_2.w):
        x = max(image_1.w, image_2.w)

    if M[1, 2] < 0:
        y = image_1.h
    else:
        y = int(image_2.h + abs(M[1, 2]))

    if y < max(image_1.h, image_2.h):
        y = max(image_1.h, image_2.h)

    output = np.zeros((y, x, 3), np.uint8)

    if orientation == "horizontal":
        image2 = image_2.image[:image_2.h, int(image_2.w*H_CROP_AMOUNT):image_2.w]
        M[0, 2] = M[0, 2] + int(image_2.w*H_CROP_AMOUNT)

    elif orientation == "vertical":
        image2 = image_2.image[int(image_2.h*V_CROP_AMOUNT):image_2.h, :image_2.w]
        M[1, 2] = M[1, 2] + int(image_2.h*V_CROP_AMOUNT)
 
    output[:image_1.h, :image_1.w] = image_1.image

    cv2.warpPerspective(image2, M, (x, y), output, borderMode=cv2.BORDER_TRANSPARENT)
    output_im = image(output)

    return output_im


def main():
    files = glob.glob(DATASET + "/*.JPG")
    images = []

    queue = Queue()
    threads = []

    for path in files:
        t = threading.Thread(target=import_worker, args=(path, queue))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    while not queue.empty():
        images.append(queue.get())

    images.sort(key=lambda x: x.path)

    rows = []

    for i in range(0, ROWS, 2):

        rows.append(images[i*COLUMNS: (i+1)*COLUMNS])

        if (i+1)*COLUMNS < len(images):
            rows.append(images[(i+1)*COLUMNS: (i+2)*COLUMNS][::-1])

    results = []

    for i in range(len(rows)):
        t = threading.Thread(target=stich_worker, args=(rows[i], queue, i))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    while not queue.empty():
        results.append(queue.get())

    results.sort(key=lambda x: x[1])

    sorted_results = [x[0] for x in results]

    result = sorted_results[0]

    for i in range(1, ROWS):
        result = stich(result, sorted_results[i], "vertical")

    cv2.imwrite(DATASET + ".JPG", result.image)


if __name__ == "__main__":
    main()
