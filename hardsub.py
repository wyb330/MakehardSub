import cv2
import os
from argparse import ArgumentParser
from utils.subtitle_utils import subtitle_captions, str2time
from image.image_utils import remove_noise_and_smooth
from image.ocr import ocr_from_file, read_from_img


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

        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        print("Frame position : {:.3f}ms".format(cap.get(cv2.CAP_PROP_POS_MSEC)))

        # Our operations on the frame come here
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # img = image_preprocess(frame)

        # 현재 frame에서 자막 추출
        sub_img = sub_image(img, rect)
        text = detect_sub_text(sub_img, lang=lang)
        print(text)

        img = draw_sub_border(img, rect)
        img = cv2.resize(img, (960, 540))

        # Display the resulting frame
        cv2.imshow(os.path.basename(args.video), img)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def make_sub(video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise Exception("video not opened")

    print("Frame count : {}".format(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("Frame per sec : {}".format(cap.get(cv2.CAP_PROP_FPS)))
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        # print("Frame position : {}".format(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        print("Frame position : {:.3f}ms".format(cap.get(cv2.CAP_PROP_POS_MSEC)))

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow(os.path.basename(args.video), gray)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main(args):
    rect = args.pos.split(',')
    rect = [int(v) for v in rect]
    print(rect)
    if args.ref:
        make_sub_with_ref(args.video, args.ref, rect, args.lang)
    else:
        make_sub(args.video)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", default="c:/tmp/5.mp4")
    parser.add_argument("--ref", default="c:/tmp/5.smi")
    parser.add_argument("--lang", default="kor")
    # parser.add_argument("--pos", default="0,800,1900,250")
    parser.add_argument("--pos", default="0,350,950,150")
    args = parser.parse_args()
    main(args)
