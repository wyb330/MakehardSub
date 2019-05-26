import cv2
import os
from argparse import ArgumentParser
from utils.subtitle_utils import subtitle_captions, str2time


def make_sub_with_ref(video, ref):
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
    for c in captions:
        s = str2time(c.start)
        e = str2time(c.end)
        pos = int((s + ((e - s) / 2)) * fps / 1000)

        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        print("Frame position : {:.3f}ms".format(cap.get(cv2.CAP_PROP_POS_MSEC)))

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (960, 540))

        # Display the resulting frame
        cv2.imshow(os.path.basename(args.video), img)
        if cv2.waitKey(2000) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # while cap.isOpened():
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #     # print("Frame position : {}".format(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    #     print("Frame position : {:.3f}ms".format(cap.get(cv2.CAP_PROP_POS_MSEC)))
    #
    #     # Our operations on the frame come here
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    #     # Display the resulting frame
    #     cv2.imshow(os.path.basename(args.video), gray)
    #     if cv2.waitKey(20) & 0xFF == ord('q'):
    #         break
    #
    # # When everything done, release the capture
    # cap.release()
    # cv2.destroyAllWindows()


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
    if args.ref:
        make_sub_with_ref(args.video, args.ref)
    else:
        make_sub(args.video)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", default="c:/tmp/34.mp4")
    parser.add_argument("--ref", default="c:/tmp/34.srt")
    args = parser.parse_args()
    main(args)
