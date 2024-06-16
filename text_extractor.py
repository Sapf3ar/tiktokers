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
    RESULT_TEXT_IDX = 1
    RESULT_CONFIDENCE_IDX = 2

    def __init__(self, **kwargs):
        self.reader = easyocr.Reader(["ru", "en"], **kwargs)

    def _get_frames(self, video_capture, sample_rate, max_frames):
        try:
            fps = video_capture.get(cv.CAP_PROP_FPS)
            measure_interval = int(fps // sample_rate)
            num_frames = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))
            frame_number = 0
            frames = []

            while video_capture.isOpened():
                ret, frame = video_capture.read()

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

    def _read_text_from_frames(self, frames, max_image_size, batch_size):
        try:
            return [
                self.reader.readtext(
                    frame, detail=1, canvas_size=max_image_size, batch_size=batch_size
                )
                for frame in frames
            ]
        except OCRException as exc:
            raise ReadTextFromFramesError from exc

    def _get_filtered_text_from_result(self, image_result, confidence_threshold):
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
        texts = []

        try:
            with video_capture(video_link) as cap:
                frames = self._get_frames(
                    video_capture=cap, sample_rate=sample_rate, max_frames=max_frames
                )

            result = self._read_text_from_frames(
                frames=frames, max_image_size=max_image_size, batch_size=batch_size
            )

            for image_result in result:
                text = self._get_filtered_text_from_result(
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

        unique_texts = list(dict.fromkeys(texts))
        video_text = " ".join(unique_texts)
        return video_text