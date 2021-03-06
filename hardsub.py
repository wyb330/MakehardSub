import cv2
import os
import glob
import time
from argparse import ArgumentParser
from utils.subtitle_utils import subtitle_captions, str2time
from utils.subtitle_utils import to_srt_timestamp, save_as_srt, save_as_smi
from subtitle.generic import Caption
from utils.file_utils import change_file_extension, extract_file_extension
from image.image_utils import image_preprocess, merge_images
from image.ocr import read_from_img
from image.text_detection import east_detect_image, east_detect_images
from skimage.measure import compare_ssim


def denoise_text(text):
    sents = text.split('\n')
    sents = [sent for sent in sents if len(sent) > 1]

    return ' '.join(sents)


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


def detect_sub_text(img, lang, t, save=False):
    if save:
        filename = os.path.join("./capture", "{}.png".format(t))
        cv2.imwrite(filename, img)
    text = read_from_img(img, lang=lang, oem=1, psm=6)
    return text


def is_sub_image(img, net):
    img = cv2.resize(img, (320, 320))
    boxes = east_detect_image(img, net)
    return True if len(boxes) > 0 else False


def detect_sub_images(images, net, min_confidence):
    images = [cv2.resize(img, (320, 320)) for img in images]
    boxes = east_detect_images(images, net, min_confidence)
    subs = [True if len(box) > 0 else False for box in boxes]

    return subs


def sub_timestamp(total_seconds):
    total_seconds = total_seconds / 1000
    hours = int(total_seconds / 3600)
    minutes = int(total_seconds / 60 - hours * 60)
    seconds = int(total_seconds - hours * 3600 - minutes * 60)
    milliseconds = round((total_seconds - seconds - hours * 3600 - minutes * 60)*1000)

    return '{:02d}_{:02d}_{:02d}_{:03d}'.format(hours, minutes, seconds, milliseconds)


def save_subtitle(output, captions):
    ext = extract_file_extension(output)
    if ext == ".smi":
        save_as_smi(output, captions)
    elif ext == ".srt":
        save_as_srt(output, captions)


def merge_sub_images(path, output):
    files = glob.glob(path)
    if len(files) > 0:
        images = []
        for file in files:
            img = cv2.imread(file)
            images.append(img)

        merge_images(images, output)
    else:
        print("합칠 이미지가 존재하지 않습니다.")


def make_sub_with_ref(video, ref, rect, lang, output, ipm=0):
    if not os.path.exists(ref):
        raise Exception("Reference subtitle error")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise Exception("video not opened")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frame count : {}".format(frame_count))
    print("Frame per sec : {}".format(fps))

    sub_captions = []
    captions = subtitle_captions(ref)
    for i, c in enumerate(captions):
        s = str2time(c.start)
        e = str2time(c.end)
        pos = int((s + ((e - s) / 2)) * fps / 1000)

        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)  # 자막 위치로 프레임 이동
        ret, frame = cap.read()

        img = frame
        if ipm == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif ipm == 2:
            img = image_preprocess(frame)
        img = cv2.resize(img, (960, 540))

        # 현재 frame에서 자막 추출
        sub_img = sub_image(img, rect)
        text = detect_sub_text(sub_img, lang=lang, t=sub_timestamp(cap.get(cv2.CAP_PROP_POS_MSEC)), save=True)
        print("[{}]Frame position : {}, {} frame".format(i,
                                                         to_srt_timestamp(cap.get(cv2.CAP_PROP_POS_MSEC)),
                                                         cap.get(cv2.CAP_PROP_POS_FRAMES)), end=" ")
        text = denoise_text(text)
        print("[{}]\n".format(text))
        caption = Caption(c.start, c.end, text)
        sub_captions.append(caption)

        img = draw_sub_border(img, rect)  # 자막 경계 박스 출력

        # Display the resulting frame
        cv2.imshow(os.path.basename(args.video), img)
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    save_subtitle(output, sub_captions)
    # merge_sub_images("./capture/*.png", "./sub_images/sub_images.png")


font = cv2.FONT_HERSHEY_SIMPLEX
# font = cv2.FONT_HERSHEY_COMPLEX
font_size = 0.5


def display_text(img, text):
    cv2.putText(img, text, (10, 20), font, font_size, (255, 255, 0), 1)


def extract_sub(not_sub):
    silence = 500
    sub_captions = []
    prior = 0
    for i in not_sub:
        if i - prior > silence:
            duration = (i - prior)
            caption = Caption(to_srt_timestamp(prior),
                              to_srt_timestamp(i),
                              'Sub duration: {:.3f}ms'.format(duration)
                              )
            sub_captions.append(caption)
        prior = i

    return sub_captions


def is_same_sub(img1, img2):
    if img2 is None:
        return False
    (score, diff) = compare_ssim(img1, img2, multichannel=True, full=True)
    if score > 0.985:
        return True
    else:
        return False


def extract_sub_frames(video, rect, frame_window, model_path, batch_size, min_confidence):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise Exception("video not opened")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fpw = fps * frame_window / 1000
    print("Frame count : {}".format(frame_count))
    print("Frame per sec : {}".format(fps))
    print("Frame per window : {}".format(fpw))

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(model_path)

    index = 0
    total_sub_frames = 0
    not_sub = []
    sub_images = []
    sub_times = []
    prev_img = (None, False)
    while cap.isOpened():
        pos = index * int(fpw)
        if pos >= frame_count:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        index += 1
        # Capture frame-by-frame
        ret, frame = cap.read()

        img = frame
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (960, 540))
        denoise_img = image_preprocess(img)
        sub_img = sub_image(img, rect)
        t = cap.get(cv2.CAP_PROP_POS_MSEC)
        if batch_size > 1:
            sub_images.append(sub_img)
            sub_times.append(t)
            if len(sub_images) >= batch_size:
                subs = detect_sub_images(sub_images, net, min_confidence)
                for i, (sub, t) in enumerate(zip(subs, sub_times)):
                    if not sub:
                        not_sub.append(t)
                    else:
                        total_sub_frames += 1
                        # cv2.imwrite(os.path.join("./capture", "{}.png".format(sub_timestamp(t))), sub_images[i])
                print("Time: {}, Number of subtitle frame : {}".format(to_srt_timestamp(t), total_sub_frames))

                sub_images = []
                sub_times = []
        else:
            if not is_same_sub(denoise_img, prev_img[0]):
                sub = is_sub_image(sub_img, net)
                if not sub:
                    not_sub.append(t)
                else:
                    # cv2.imshow("denoise image", img)
                    prev_img = (denoise_img, sub)
            else:
                if not prev_img[1]:
                    not_sub.append(t)

        img = draw_sub_border(img, rect)  # 자막 경계 박스 출력
        display_text(img, "Frame position : {}".format(to_srt_timestamp(t)))
        cv2.imshow(os.path.basename(args.video), img)
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    if len(sub_images) > 0:
        subs = detect_sub_images(sub_images, net, min_confidence)
        for sub, t in zip(subs, sub_times):
            if sub:
                not_sub.append(t)

    return not_sub


def make_sub(video, rect, frame_window, model_path, batch_size, min_confidence):
    start = time.time()

    # 영상에서 자막이 검출되지 않은 타임 코드를 추출한다.
    not_sub = extract_sub_frames(video, rect, frame_window, model_path, batch_size, min_confidence)

    # 자막이 없는 타임 코드를 이용해 자막의 타임 코드를 추출한다.
    captions = extract_sub(not_sub)

    # 타임코드 자막을 srt 형식으로 저장
    sub_file = change_file_extension(video, '.srt')
    save_as_srt(sub_file, captions)

    end = time.time()
    print("[INFO] text detection took {:.6f} seconds".format(end - start))


def main(args):
    rect = args.pos.split(',')
    rect = [int(v) for v in rect]
    if args.ref:
        if not args.output:
            raise Exception("출력 자막 파일명을 입력하세요.(--output 옵션)")
        make_sub_with_ref(args.video, args.ref, rect, args.lang, args.output, args.ipm)
    else:
        make_sub(args.video, rect, args.frame_window, args.model_path, args.batch_size, args.min_confidence)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--ref", help="타임코드 자막")
    parser.add_argument("--output", help="출력 자막 파일명")
    parser.add_argument("--lang", default="eng", help="자막 언어")
    parser.add_argument("--frame_window", default=250, type=int, help="프레임 간격(ms)")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--pos", default="240,400,500,130", help="자막 영역 좌표")
    parser.add_argument("--model_path", default="./model/frozen_east_text_detection.pb")
    parser.add_argument("--ipm", default=0, type=int, help="이미지 전처리 모드")
    parser.add_argument("--min_confidence", default=0.92, help="텍스트 확신 확률")
    args = parser.parse_args()
    main(args)
