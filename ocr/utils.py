from contextlib import contextmanager

import cv2 as cv


@contextmanager
def video_capture(filepath):
    cap = cv.VideoCapture(filepath)

    try:
        yield cap
    finally:
        cap.release()
