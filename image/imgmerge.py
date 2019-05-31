'''
python 패키지 설치

pip install opencv-python
pip install numpy
'''
import glob
import os
import cv2
import numpy as np


def merge_images(images, output):
    result = np.concatenate(images, axis=0)
    cv2.imwrite(output, result)


def main(path, save_path, batch_size=100):
    files = glob.glob(path)
    if len(files) > 0:
        images = []
        index = 0
        for file in files:
            img = cv2.imread(file)
            images.append(img)
            index += 1
            if len(images) >= batch_size:
                merge_images(images, os.path.join(save_path, "sub_images{}.png".format(index)))
                images = []

        if len(images) > 0:
            merge_images(images, os.path.join(save_path, "sub_images{}.png".format(index)))
    else:
        print("합칠 이미지가 존재하지 않습니다.")


if __name__ == "__main__":
    main("C:/videosubfinder/RGBImages/*.jpeg", "C:/videosubfinder/TXTImages", 100)
