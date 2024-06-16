from contextlib import contextmanager

import cv2 as cv
from exceptions import GetVideoCaptureError, YieldVideoCaptureError


@contextmanager
def video_capture(filepath):
    try:
        cap = cv.VideoCapture(filepath)
    except Exception as exc:
        raise GetVideoCaptureError from exc
    else:
        try:
            yield cap
        except Exception as exc:
            raise YieldVideoCaptureError from exc
        finally:
            cap.release()
