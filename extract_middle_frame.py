import argparse
import itertools
import pdb

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-video", required=True, help="path to input video")
    parser.add_argument("-o", "--output-path", required=True, help="path of output")
    args = parser.parse_args()

    video_capture = cv2.VideoCapture(args.input_video)
    frame_index = video_capture.get(cv2.CAP_PROP_FRAME_COUNT) // 2

    for i in itertools.count():
        _, image = video_capture.read()
        if i == frame_index:
            cv2.imwrite(args.output_path, image)
            break
        i += 1


if __name__ == "__main__":
    main()
