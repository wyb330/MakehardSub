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
from utils.subtitle_utils import save_as_srt, save_as_smi
from utils.file_utils import extract_file_extension
from subtitle.subtitle import Subtitle


def save_subtitle(file, captions):
    ext = extract_file_extension(file)
    if ext == ".smi":
        save_as_smi(file, captions)
    elif ext == ".srt":
        save_as_srt(file, captions)


font = cv2.FONT_HERSHEY_TRIPLEX
font_size = 1.2


def display_text(img, text):
    cv2.putText(img, text, (30, 40), font, font_size, (0, 0, 0), thickness=2)


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
    자막이 없는 영역은 흰색으로 채워져 있으므로 영역의 평균 및 표준편차 색상값으로 자막인지 여부를 판별한다.
    :param image: 자막이 있는 이미지
    :return:
    """
    (H, W) = image.shape[:2]
    index = 0
    d = 5
    threshold = 240

    while (index + 1) * d < H:
        img = sub_image(image, [0, index * d, W, d])
        avg = np.mean(img)
        std = np.std(img)
        if (avg < threshold) and (std > 10):
            break
        index += 1

    return W, H, index * d


def main(path, save_path, pos, batch_size=100, output=None):
    rect = pos.split(',')
    rect = [int(v) for v in rect]
    files = glob.glob(path)
    h, w = (0, 0)
    if len(files) > 0:
        # 자막 영역이 지정되어 있지 않으면 자막 영역을 찾는다.
        if rect[2] == 0:
            width, height, top = detect_sub_area(cv2.imread(files[0]))
            h_t = 40 * 2  # 타임 코드 텍스트 높이
            rect = [0, top - h_t, width, height - top + h_t]

        images = []
        index = 0
        subtitle = Subtitle()
        for file in tqdm.tqdm(files):
            img = cv2.imread(file)
            img = sub_image(img, rect)

            sub_t = '.'.join(os.path.basename(file).split('.')[:-1])
            if sub_t.endswith('!'):
                sub_t = str(sub_t[:-1])
            sub_t = sub_t.split('__')
            start = to_srt_timestamp(str2time(sub_t[0]))
            end = to_srt_timestamp(str2time(sub_t[1]))
            display_text(img, "{} --> {}".format(start, end))
            duration = str2time(sub_t[1]) - str2time(sub_t[0])
            subtitle.add(start, end, 'Sub duration: {:.3f}ms'.format(duration))

            if index == 0:
                h, w = img.shape[:2]
            else:
                h_n, w_n = img.shape[:2]
                if (w != w_n) or (h != h_n):
                    img = cv2.resize(img, (w, h))
            images.append(img)
            index += 1
            if len(images) >= batch_size:
                merge_images(images, os.path.join(save_path, "sub_images{}.png".format(index)))
                images = []

        if len(images) > 0:
            merge_images(images, os.path.join(save_path, "sub_images{}.png".format(index)))

        # 타임코드용 자막 저장
        if output is not None:
            save_subtitle(os.path.join(save_path, output), subtitle)
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
    parser.add_argument("-s", default="sub.srt", type=str, help="저장할 타임코드용 자막")
    args = parser.parse_args()
    main(args.i, args.o, args.r, args.b, args.s)
