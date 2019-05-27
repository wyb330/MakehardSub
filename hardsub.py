import cv2
import os
from argparse import ArgumentParser
from utils.subtitle_utils import subtitle_captions, str2time
from image.image_utils import remove_noise_and_smooth
from image.ocr import read_from_img


def image_preprocess(img):
    height, width, _ = img.shape
    img = remove_noise_and_smooth(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def draw_sub_border(img, rect):
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 1)

    return img


def sub_image(img, rect):
    column = rect[0]
    row = rect[1]
    width = rect[2]
    height = rect[3]
    img = img[row:row+height, column:column+width]
    return img


def detect_sub_text(img, lang):
    # filename = os.path.join("./capture", "{}.png".format(i))
    # cv2.imwrite(filename, sub_img)
    # text = ocr_from_file(filename)
    text = read_from_img(img, lang=lang)
    return text


def is_sub_image(img, lang):
    # text = read_from_img(img, lang=lang)
    # if text:
    #     return True
    # else:
    #     return False
    return False


def make_sub_with_ref(video, ref, rect, lang):
    if not os.path.exists(ref):
        raise Exception("Reference subtitle error")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise Exception("video not opened")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frame count : {}".format(frame_count))
    print("Frame per sec : {}".format(fps))

    captions = subtitle_captions(ref)
    for i, c in enumerate(captions):
        s = str2time(c.start)
        e = str2time(c.end)
        pos = int((s + ((e - s) / 2)) * fps / 1000)

        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)  # 자막 위치로 프레임 이동
        ret, frame = cap.read()
        print("Frame position : {:.3f}ms, {} frame".format(cap.get(cv2.CAP_PROP_POS_MSEC), cap.get(cv2.CAP_PROP_POS_FRAMES)))

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # img = image_preprocess(frame)

        # 현재 frame에서 자막 추출
        sub_img = sub_image(img, rect)
        text = detect_sub_text(sub_img, lang=lang)
        print(text)

        img = draw_sub_border(img, rect)  # 자막 경계 박스 출력
        img = cv2.resize(img, (960, 540))

        # Display the resulting frame
        cv2.imshow(os.path.basename(args.video), img)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


font = cv2.FONT_HERSHEY_SIMPLEX
# font = cv2.FONT_HERSHEY_COMPLEX
font_size = 0.5


def display_text(img, text):
    cv2.putText(img, text, (10, 20), font, font_size, (255, 255, 0), 1)


def make_sub(video, rect, lang):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise Exception("video not opened")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_window = 250
    fpw = fps * frame_window / 1000
    print("Frame count : {}".format(frame_count))
    print("Frame per sec : {}".format(fps))
    print("Frame per window : {}".format(fpw))

    index = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        print("Frame position : {:.3f}ms, {} frame".format(cap.get(cv2.CAP_PROP_POS_MSEC), cap.get(cv2.CAP_PROP_POS_FRAMES)))

        # Our operations on the frame come here
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # img = image_preprocess(frame)
        img = cv2.resize(img, (960, 540))
        sub_img = sub_image(img, rect)
        sub = is_sub_image(sub_img, lang=lang)
        display_text(sub_img, "Subtitle Frame : {}".format(sub))
        img = draw_sub_border(img, rect)  # 자막 경계 박스 출력

        # Display the resulting frame
        cv2.imshow(os.path.basename(args.video), img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        pos = index * fpw
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        index += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main(args):
    rect = args.pos.split(',')
    rect = [int(v) for v in rect]
    if args.ref:
        make_sub_with_ref(args.video, args.ref, rect, args.lang)
    else:
        make_sub(args.video, rect, args.lang)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", default="c:/tmp/5.mp4")
    parser.add_argument("--ref", default=None)
    parser.add_argument("--lang", default="kor")
    parser.add_argument("--pos", default="5,380,950,150")
    args = parser.parse_args()
    main(args)
