import math
from typing import List

import numpy as np
import requests
import torch
from decord import VideoReader, cpu
from llavavid.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from llavavid.conversation import SeparatorStyle, conv_templates
from llavavid.mm_utils import (KeywordsStoppingCriteria,
                               get_model_name_from_path, tokenizer_image_token)
from llavavid.model.builder import load_pretrained_model
from transformers import AutoConfig


def split_list(lst: list, n: int) -> List[list]:
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst: list, n: int, k: int) -> list:
    chunks = split_list(lst, n)
    return chunks[k]


def load_video(video_path: str, for_get_frames_num: int) -> np.ndarray:
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames


def download_file(url: str, path: str, timeout: int = 60):
    try:
        with open(path, 'wb') as f:
            f.write(requests.get(url, timeout=timeout).content)
    except Exception as err:
        print(url, err)
        print('retrying')
        download_file(url, path, timeout=timeout+30)


class VideoCaptioner:
    """VideoCaptioner is a class used for creating captions for videos using LLaVA-Next model

    Parameters
    ----------
    device: torch.device | str
        device that is used for model

    Attributes
    ----------
    device: torch.device | str
        device that is used for model

    Methods
    -------
    get_caption(video_path: str)
        Generates a caption for video
    """


    def __init__(self, device: str):
        self.device = device
        self.model_path = "lmms-lab/LLaVA-NeXT-Video-7B-DPO"
        self.model_base = None
        self.conv_mode = "vicuna_v1"
        model_name = get_model_name_from_path(self.model_path)

        self.mm_spatial_pool_stride = 2
        self.for_get_frames_num = 32

        self.overwrite_config = {}
        self.overwrite_config["mm_resampler_type"] = "spatial_pool"
        self.overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        self.overwrite_config["mm_spatial_pool_out_channels"] = 1024
        self.overwrite_config["mm_spatial_pool_mode"] = "average"
        self.overwrite_config["patchify_video_feature"] = False

        cfg_pretrained = AutoConfig.from_pretrained(self.model_path)

        if "224" in cfg_pretrained.mm_vision_tower:
            # suppose the length of text tokens is around 1000, from bo's report
            least_token_number = self.for_get_frames_num*(16//self.mm_spatial_pool_stride)**2 + 1000
        else:
            least_token_number = self.for_get_frames_num*(24//self.mm_spatial_pool_stride)**2 + 1000

        self.scaling_factor = math.ceil(least_token_number/4096)

        if self.scaling_factor >= 2:
            if "mistral" not in cfg_pretrained._name_or_path.lower() and "7b" in cfg_pretrained._name_or_path.lower():
                self.overwrite_config["rope_scaling"] = {"factor": float(self.scaling_factor), "type": "linear"}
            self.overwrite_config["max_sequence_length"] = 4096 * self.scaling_factor
            self.overwrite_config["tokenizer_model_max_length"] = 4096 * self.scaling_factor

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path,
            self.model_base,
            model_name,
            load_8bit=False,
            overwrite_config=self.overwrite_config
        )

    def get_caption(self, video_path: str) -> str:
        """Generates a caption.

        Parameters
        ----------
        video_path: str
            Path to video file

        Returns
        -------
        str:
            Generated caption
        """
        question = "Please provide a detailed description of the video, actions in the video and background scenes."

        sample_set = {}
        sample_set["Q"] = question
        sample_set["video_name"] = video_path

        video = load_video(video_path, self.for_get_frames_num)
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().to(self.device)
        video = [video]

        qs = question
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().to(self.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        cur_prompt = question
        with torch.inference_mode():
            self.model.update_prompt([[cur_prompt]])
            output_ids = self.model.generate(
                inputs=input_ids, images=video, attention_mask=attention_masks,
                modalities="video", do_sample=True, temperature=0.2, max_new_tokens=512,
                use_cache=True, stopping_criteria=[stopping_criteria]
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs
