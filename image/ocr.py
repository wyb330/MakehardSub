"""
이미지에 있는 텍스트를 추출한다.
pytesseract는 shell 명령어를 이용하고 pyocr은 shell과 dll 모두 사용가능함.
속도 문제 때문에 pyocr에서 tesseract dll를 사용할 수 있으면, 이를 먼저 사용하고
그렇지 않다면 pytesseract를 사용함

tesseract 한글 데이터 다운로드 https://github.com/tesseract-ocr/tessdata
"""
import numpy as np
import pytesseract
import pyocr
import pyocr.builders
from PIL import Image
import cv2


enlarge = 1

ocr_tools = pyocr.get_available_tools()
if len(ocr_tools) == 0:
    print("No OCR tool found")
else:
    # The tools are returned in the recommended order of usage
    tool = ocr_tools[0]
    # print("Will use tool '%s'" % (tool.get_name()))


def adjust_image(img):
    if len(img) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width, channel = img_gray.shape
    else:
        img_gray = img
        height, width = img.shape
    gray_enlarge = cv2.resize(img_gray, (enlarge * width, enlarge * height), interpolation=cv2.INTER_LINEAR)

    # Denoising
    denoised = cv2.fastNlMeansDenoising(gray_enlarge, h=10, searchWindowSize=21, templateWindowSize=7)

    # binary 이미지로 변환
    gray_pin = 196
    ret, thresh = cv2.threshold(denoised, gray_pin, 255, cv2.THRESH_BINARY)

    return thresh


def image2string(img):
    return tool.image_to_string(Image.fromarray(img), lang="eng", builder=pyocr.builders.TextBuilder())


def find_text_region(img):
    line_and_word_boxes = tool.image_to_string(Image.fromarray(img), lang="eng", builder=pyocr.builders.LineBoxBuilder())
    text_regions = []
    for line_box in line_and_word_boxes:
        print(line_box.content, line_box.position)
        region = (line_box.position[0][0],
                  line_box.position[0][1],
                  line_box.position[1][0],
                  line_box.position[1][1]
                  )
        text_regions.append(region)

    final_regions = []
    if (len(text_regions) > 0):
        final_regions = text_regions
        final_regions = np.array(final_regions)
        area = 0
        x = 0
        for r in text_regions:
            if r[2] * r[3] > area:
                area = r[2] * r[3]
                x = r[0]
        y = 10000
        for r in text_regions:
            if (r[1] < y) and (r[0] == x):
                y = r[1]
        w = np.max(final_regions[:, 2])
        y_m = np.max(final_regions[:, 1])
        h_m = np.max(final_regions[:, 3])
        w = w
        h = h_m
        final_regions = [x, y, w, h]
    return final_regions


# def ocr_from_array(img):
#     img = adjust_image(img)
#     ocr_text = tool.image_to_string(Image.fromarray(img),
#                                     lang="eng",
#                                     builder=pyocr.builders.TextBuilder())
#
#     return ocr_text


def ocr_from_array(img):
    img = adjust_image(img)
    ocr_text = pytesseract.image_to_string(Image.fromarray(img),
                                           lang="kor")

    return ocr_text


def ocr_from_file(file):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    return ocr_from_array(img)


def region_from_file(file):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = adjust_image(img)
    return find_text_region(img)


def ocr_fom_file(file):
    img = cv2.imread(file)
    return ocr_from_array(img)


def read_from_img(img):
    text = ocr_from_array(img)
    return text


def read_from_file(file):
    if len(ocr_tools) > 0:
        text = ocr_from_file(file)
    # else:
    #     text = ocr_fom_file(file)

    return text


def read_text_from_image(img, rg, enlarge_ratio=1.0):
    h, w = img.shape[:2]
    if len(rg) == 0:
        text_img = img
    else:
        x = rg[0]
        y = rg[1]
        end_x = rg[2]
        end_y = rg[3]
        text_img = img[y: end_y, x: end_x]

    if enlarge_ratio > 1:
        text_img = cv2.resize(text_img, (int(w * enlarge_ratio), int(h * enlarge_ratio)))
    return image2string(text_img)


if __name__ == "__main__":
    img_path = "../capture/2.png"

    text = ocr_from_file(img_path)
    sents = text.split('\n')
    sents = [sent for sent in sents if sent]
    for sent in sents:
        print(sent)

    # region = region_from_file(img_path)
    # print(region)
