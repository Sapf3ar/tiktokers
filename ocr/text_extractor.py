"""A module for text extraction from the video."""

import cv2 as cv
import easyocr
from exceptions import (
    FilterTextsFromResultError,
    GetFramesError,
    GetVideoCaptureError,
    OCRException,
    ReadTextFromFramesError,
    YieldVideoCaptureError,
)
from loguru import logger
from utils import video_capture


class TextExtractor:
    """
    A class used to extract text from video frames using Optical Character Recognition (OCR).

    Attributes
    ----------
    reader : easyocr.Reader
        An instance of the EasyOCR reader for text detection and extraction.

    Methods
    -------
    get_frames(video_capture, sample_rate, max_frames)
        Extracts frames from the video at a specified sample rate.
    read_text_from_frames(frames, max_image_size, batch_size)
        Reads and extracts text from the given frames.
    get_filtered_text_from_result(image_result, confidence_threshold)
        Filters the extracted text based on a confidence threshold.
    extract_text(video_link, sample_rate, confidence_threshold,
                 max_image_size, max_frames, batch_size)
        Extracts text from a video using the above methods.
    """

    RESULT_TEXT_IDX = 1
    RESULT_CONFIDENCE_IDX = 2

    def __init__(self, **kwargs):
        """
        Constructs all the necessary attributes for the TextExtractor object.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments for the EasyOCR reader.
        """
        self.reader = easyocr.Reader(["ru", "en"], **kwargs)

    def get_frames(self, video_cap, sample_rate, max_frames):
        """
        Extracts frames from the video at a specified sample rate.

        Parameters
        ----------
        video_cap : cv.VideoCapture
            The video capture object.
        sample_rate : float
            The rate at which frames should be sampled from the video.
        max_frames : int
            The maximum number of frames to extract from the video.

        Returns
        -------
        list
            A list of extracted frames.

        Raises
        ------
        GetFramesError
            If there is an error in getting frames from the video capture.
        """
        try:
            fps = video_cap.get(cv.CAP_PROP_FPS)
            measure_interval = int(fps // sample_rate)
            num_frames = int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))
            frame_number = 0
            frames = []

            while video_cap.isOpened():
                ret, frame = video_cap.read()

                if not ret or frame_number > num_frames or len(frames) > max_frames:
                    break

                frame_number += 1

                if frame_number % measure_interval != 0:
                    continue

                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                frames.append(frame)
            return frames
        except OCRException as exc:
            raise GetFramesError from exc

    def read_text_from_frames(self, frames, max_image_size, batch_size):
        """
        Reads and extracts text from the given frames.

        Parameters
        ----------
        frames : list
            The list of frames to extract text from.
        max_image_size : int
            The maximum size of the image for the OCR process.
        batch_size : int
            The number of frames to process at once.

        Returns
        -------
        list
            A list of results from the OCR process.

        Raises
        ------
        ReadTextFromFramesError
            If there is an error in reading text from the frames.
        """
        try:
            return [
                self.reader.readtext(
                    frame, detail=1, canvas_size=max_image_size, batch_size=batch_size
                )
                for frame in frames
            ]
        except OCRException as exc:
            raise ReadTextFromFramesError from exc

    def get_filtered_text_from_result(self, image_result, confidence_threshold):
        """
        Filters the extracted text based on a confidence threshold.

        Parameters
        ----------
        image_result : list
            The result from the OCR process.
        confidence_threshold : float
            The confidence threshold for the OCR process.

        Returns
        -------
        str
            The filtered text.

        Raises
        ------
        FilterTextsFromResultError
            If there is an error in filtering the recognized texts.
        """
        try:
            text = " ".join(
                [
                    x[self.RESULT_TEXT_IDX]
                    if x[self.RESULT_CONFIDENCE_IDX] > confidence_threshold
                    else ""
                    for x in image_result
                ]
            )
            return text.strip()
        except OCRException as exc:
            raise FilterTextsFromResultError from exc

    def extract_text(
        self,
        video_link: str,
        sample_rate: float = 0.6,
        confidence_threshold: float = 0.9,
        max_image_size: int = 400,
        max_frames: int = 25,
        batch_size: int = 16,
    ) -> list[str]:
        """
        Extracts text from a video using the above methods.

        Parameters
        ----------
        video_link : str
            The link to the video.
        sample_rate : float, optional
            The rate at which frames should be sampled from the video (default is 0.6).
        confidence_threshold : float, optional
            The confidence threshold for the OCR process (default is 0.9).
        max_image_size : int, optional
            The maximum size of the image for the OCR process (default is 400).
        max_frames : int, optional
            The maximum number of frames to extract from the video (default is 25).
        batch_size : int, optional
            The number of frames to process at once (default is 16).

        Returns
        -------
        list
            A list of extracted texts.

        Raises
        ------
        GetFramesError
            If there is an error in getting frames from the video capture.
        ReadTextFromFramesError
            If there is an error in reading text from the frames.
        FilterTextsFromResultError
            If there is an error in filtering the recognized texts.
        GetVideoCaptureError
            If there is an error in getting video capture from the given path.
        YieldVideoCaptureError
            If there is an error in yielding video capture.
        """
        texts = []

        try:
            with video_capture(video_link) as cap:
                frames = self.get_frames(
                    video_cap=cap, sample_rate=sample_rate, max_frames=max_frames
                )

            result = self.read_text_from_frames(
                frames=frames, max_image_size=max_image_size, batch_size=batch_size
            )

            for image_result in result:
                text = self.get_filtered_text_from_result(
                    image_result=image_result, confidence_threshold=confidence_threshold
                )
                if text:
                    texts.append(text)

        except GetFramesError:
            logger.exception("Error when getting frames from video capture")
        except ReadTextFromFramesError:
            logger.exception("Error when reading text from frames")
        except FilterTextsFromResultError:
            logger.exception("Error when filtering recognized texts")
        except GetVideoCaptureError:
            logger.exception("Error when getting video capture from given path")
        except YieldVideoCaptureError:
            logger.exception("Error when yielding video capture")

        return texts
