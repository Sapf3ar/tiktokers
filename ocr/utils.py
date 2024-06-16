"""
This module provides a context manager for handling video capture in OpenCV.

The context manager, `video_capture`, simplifies the process of opening and
closing video captures. It ensures that the video capture is properly released,
even if an error occurs during processing.

Functions
---------
video_capture(filepath)
    A context manager for handling video capture in OpenCV.
"""

from contextlib import contextmanager

import cv2 as cv
from exceptions import GetVideoCaptureError, YieldVideoCaptureError


@contextmanager
def video_capture(filepath):
    """
    A context manager for handling video capture in OpenCV.

    This context manager simplifies the process of opening and closing video captures.
    It ensures that the video capture is properly released,
    even if an error occurs during processing.

    Parameters
    ----------
    filepath : str
        The path to the video file.

    Yields
    ------
    cv.VideoCapture
        The video capture object.

    Raises
    ------
    GetVideoCaptureError
        If there is an error in getting video capture from the given path.
    YieldVideoCaptureError
        If there is an error in yielding video capture.
    """
    try:
        cap = cv.VideoCapture(filepath)
    except Exception as exc:
        raise GetVideoCaptureError from exc

    try:
        yield cap
    except Exception as exc:
        raise YieldVideoCaptureError from exc
    finally:
        cap.release()
