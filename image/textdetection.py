#!/usr/bin/python

import os
import cv2
import numpy as np


classifierNM1 = 'trained_classifierNM1.xml'
classifierNM2 = 'trained_classifierNM2.xml'


def find_text_region(image, path, show_edge=False):
    # for visualization
    vis = image.copy()

    # Extract channels to be processed individually
    channels = cv2.text.computeNMChannels(image)
    # Append negative channels to detect ER- (bright regions over dark background)
    cn = len(channels) - 1
    for c in range(0, cn):
        channels.append((255 - channels[c]))

    # Apply the default cascade classifier to each independent channel (could be done in parallel)
    text_regions = []
    for i, channel in enumerate(channels):
        erc1 = cv2.text.loadClassifierNM1(os.path.join(path, classifierNM1))
        er1 = cv2.text.createERFilterNM1(erc1, 16, 0.00015, 0.13, 0.2, True, 0.1)

        erc2 = cv2.text.loadClassifierNM2(os.path.join(path, classifierNM2))
        er2 = cv2.text.createERFilterNM2(erc2, 0.5)

        regions = cv2.text.detectRegions(channel, er1, er2)
        if len(regions) == 0:
            continue
        rects = cv2.text.erGrouping(image, channel, [r.tolist() for r in regions])

        # Visualization
        for r in range(0, np.shape(rects)[0]):
            rect = rects[r]
            # 일정 크기 이하의 영역은 오류로 간주하고 무시한다.
            if rect[2] < 60:
                continue
            text_regions.append(rect)
            if show_edge:
                cv2.rectangle(vis, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 1)

    if (len(text_regions) > 0) and (len(text_regions) > 0):
        final_regions = text_regions
        final_regions = np.array(final_regions)
        area = 0
        x = 0
        for r in text_regions:
            if r[2] * r[3] > area:
                area = r[2] * r[3]
                x = r[0]
        # x = np.min(final_regions[:, 0])
        y = 10000
        for r in text_regions:
            if (r[1] < y) and (r[0] == x):
                y = r[1]
        # y = np.min(final_regions[:, 1])
        w = np.max(final_regions[:, 2])
        # h = np.max(final_regions[:, 3])
        y_m = np.max(final_regions[:, 1])
        h_m = np.max(final_regions[:, 3])
        w = x + w
        h = y_m + h_m
        final_regions = [x, y, w, h]

        if show_edge:
            cv2.rectangle(vis, (x, y), (w, h), (0, 0, 255), 2)
    else:
        final_regions = []

    return final_regions, vis


if __name__ == "__main__":
    img = cv2.imread('../capture/3.png')
    rg, img = find_text_region(img, './', True)
    img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
    cv2.imshow("Text detection result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
