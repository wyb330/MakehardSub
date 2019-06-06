'''
python 패키지 설치

pip install opencv-python
pip install numpy
pip install tqdm
'''
import glob
import os
import cv2
import numpy as np
from argparse import ArgumentParser
import tqdm


font = cv2.FONT_HERSHEY_TRIPLEX
font_size = 1.2


def display_text(img, text):
    cv2.putText(img, text, (30, 50), font, font_size, (0, 0, 0), thickness=2)


def merge_images(images, output):
    result = np.concatenate(images, axis=0)
    cv2.imwrite(output, result)


def to_srt_timestamp(total_seconds):
    total_seconds = total_seconds / 1000
    hours = int(total_seconds / 3600)
    minutes = int(total_seconds / 60 - hours * 60)
    seconds = int(total_seconds - hours * 3600 - minutes * 60)
    milliseconds = round((total_seconds - seconds - hours * 3600 - minutes * 60)*1000)

    return '{:02d}:{:02d}:{:02d}.{:03d}'.format(hours, minutes, seconds, milliseconds)


def str2time(time):
    t = time.split('_')
    seconds = int(t[0]) * 3600 + 60 * int(t[1]) + int(t[2]) + (int(t[3]) / 1000)
    return int(seconds * 1000)


def sub_image(img, rect):
    column = rect[0]
    row = rect[1]
    width = rect[2]
    height = rect[3]
    img = img[row:row+height, column:column+width]
    return img


def detect_sub_area(image):
    """
    이미지에서 자막이 있는 영역을 찾는다.
    자막이 없는 영역은 흰색으로 채워져 있으므로 영역의 평균 색상값으로 자막인지 여부를 판별한다.
    :param image: 자막이 있는 이미지
    :return:
    """
    (H, W) = image.shape[:2]
    index = 0
    d = 10
    threshold = 165

    while (index + 1) * d < H:
        img = sub_image(image, [0, index * d, W, (index + 1) * d])
        avg = np.mean(img)
        if avg < threshold:
            break
        index += 1

    return W, H, index * 10


def main(path, save_path, pos, batch_size=100):
    rect = pos.split(',')
    rect = [int(v) for v in rect]
    files = glob.glob(path)
    if len(files) > 0:
        images = []
        index = 0
        for file in tqdm.tqdm(files):
            img = cv2.imread(file)
            if rect[2] > 0:
                img = sub_image(img, rect)
            else:
                width, height, top = detect_sub_area(img)
                r = [0, top, width, height - top]
                img = sub_image(img, r)
            sub_t = '.'.join(os.path.basename(file).split('.')[:-1])
            if sub_t.endswith('!'):
                sub_t = str(sub_t[:-1])
            sub_t = sub_t.split('__')
            start = to_srt_timestamp(str2time(sub_t[0]))
            end = to_srt_timestamp(str2time(sub_t[1]))
            display_text(img, "{} --> {}".format(start, end))
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
    parser = ArgumentParser()
    parser.add_argument("-i", help="자막 이미지", required=True)
    parser.add_argument("-o", help="저장할 디렉토리", required=True)
    parser.add_argument("-b", default=50, type=int, help="합칠 이미지 단위")
    # 자막 영역 좌표 - left,top,width,height
    # 영역 지정을 하지 않으려면 width을 0으로 설정
    parser.add_argument("-r", default="0,0,0,0", type=str, help="자막 영역 좌표")
    args = parser.parse_args()
    main(args.i, args.o, args.r, args.b)
