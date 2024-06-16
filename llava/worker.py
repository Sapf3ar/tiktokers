import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import requests

from .logger import get_logger
from .video_captioner import VideoCaptioner, download_file


class DebugCaptioner:

    def get_caption(self, video_path: str) -> str:
        return "test caption"


@dataclass
class WorkerConfig:
    """Config for Worker class"""
    worker_id: str
    host: str
    sleep_time: float = 5
    timeout: float = 15


class Worker:
    """VideoCaptioner is a class used for creating captions for videos using LLaVA-Next model

    Parameters
    ----------
    config: WorkerConfig
        Config for worker
    debug: bool = False
        Run worker in debug mode. It will not use model and return dummy captions

    Attributes
    ----------
    config: WorkerConfig
        device that is used for model
    captioner: VideoCaptioner
        Class for caption generation
    logger: logging.Logger
        Logger class instance for logging
    """

    def __init__(self, config: WorkerConfig, debug: bool = False):
        self.config = config
        if debug:
            self.captioner = DebugCaptioner()
        else:
            self.captioner = VideoCaptioner("cuda")

        os.makedirs('logs', exist_ok=True)
        self.logger = get_logger(os.path.join('logs', self.config.worker_id + '.log'))

    def get_task(self) -> Dict[str, Any]:
        """Get the task from server

        Returns
        -------
        Dict[str, Any]:
            Dict with task data
        """
        result = requests.get(
            self.config.host + '/get_task',
            timeout=self.config.timeout
        ).json()
        return result

    def send_result(self, result: Dict[str, Any], retries: int = 3) -> bool:
        """Send result to server

        Parameters
        ----------
        result: Dict[str, Any]
            Jsonable dict with task result
        retries: int = 3
            Number of retries

        Returns
        -------
        bool:
            Whether the request to server was successfull
        """
        ok = False
        for retry in range(retries):
            try:
                requests.post(
                    self.config.host + '/push_last_task',
                    json=result,
                    timeout=self.config.timeout
                )
            except Exception:
                self.logger.exception(f'Error during sending result. Retry {retry}/{retries}')
            else:
                self.logger.info('result sended successfully')
                ok = True
                break
        return ok

    def process_task(self, link: str) -> Tuple[str, str, float]:
        """Send result to server

        Parameters
        ----------
        link: str
            Link to the video

        Returns
        -------
        Tuple[str, str, float]:
            Generated caption, error and execution time
        """
        t0 = time.time()
        try:
            os.makedirs('videos', exist_ok=True)
            path = os.path.join('videos', uuid.uuid4().hex + '.' + link.split('.')[-1])
            download_file(link, path)
            caption = self.captioner.get_caption(path)
            os.remove(path)
            execution_time = time.time()-t0
            return caption, "Success", execution_time
        except Exception as err:
            self.logger.exception('Error during process task')
            execution_time = time.time()-t0
            return "", str(err), execution_time

    def run_loop(self):
        """Run an endless loop for processing tasks"""
        while True:
            time.sleep(self.config.sleep_time)
            self.logger.debug("Getting new task")
            try:
                task = self.get_task()
            except Exception:
                self.logger.exception('Error during getting task')
                time.sleep(self.config.sleep_time)
                continue

            if task == -1:
                self.logger.debug("No new tasks")
                continue

            self.logger.info(f"Processing task: {task}")
            caption, error, exec_time = self.process_task(task["link"])
            result = {
                "id_": int(task["task_id"]),
                "execution_time": int(exec_time),
                "video_link": task["link"],
                "caption": caption,
                "error": error
            }
            self.logger.info(f"Sending result: {result}")
            self.send_result(result)
