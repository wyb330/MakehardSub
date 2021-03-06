"""
tesseract 한글 데이터 다운로드 https://github.com/tesseract-ocr/tessdata
"""
import pytesseract
from PIL import Image
import cv2
from image.image_utils import image_preprocess


enlarge = 1


def adjust_image(img):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img_gray.shape
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


def ocr_from_array(img, lang, oem=1, psm=6):
    '''
    oem
      0: Legacy Engine only
      1: LSTM Engine only
      2: Legacy and LSTM engine

    Page segmentation modes:
      0    Orientation and script detection (OSD) only.
      1    Automatic page segmentation with OSD.
      2    Automatic page segmentation, but no OSD, or OCR.
      3    Fully automatic page segmentation, but no OSD. (Default)
      4    Assume a single column of text of variable sizes.
      5    Assume a single uniform block of vertically aligned text.
      6    Assume a single uniform block of text.
      7    Treat the image as a single text line.
      8    Treat the image as a single word.
      9    Treat the image as a single word in a circle.
     10    Treat the image as a single character.
     11    Sparse text. Find as much text as possible in no particular order.
     12    Sparse text with OSD.
     13    Raw line. Treat the image as a single text line,
           bypassing hacks that are Tesseract-specific.

    :param img:
    :param lang:
    :return:
    '''
    # img = adjust_image(img)
    config = ('--oem {}  --psm {}'.format(oem, psm))
    ocr_text = pytesseract.image_to_string(Image.fromarray(img),
                                           config=config,
                                           lang=lang)

    return ocr_text


def ocr_from_file(file, lang):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    return ocr_from_array(img, lang)


def ocr_fom_file(file, lang):
    img = cv2.imread(file)
    return ocr_from_array(img, lang)


def read_from_img(img, lang, oem=1, psm=6):
    text = ocr_from_array(img, lang, oem, psm)
    return text


def read_from_file(file, lang):
    text = ocr_fom_file(file, lang)

    return text


if __name__ == "__main__":
    img_path = "../capture/17.png"

    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = image_preprocess(img)
    text = read_from_img(img, lang="kor", oem=1, psm=6)
    sents = text.split('\n')
    sents = [sent for sent in sents if sent]
    for sent in sents:
        print(sent)
    cv2.imshow("image", img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
