"""Flux inpainting model using the diffusers implementation."""

import os
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import FluxInpaintPipeline

from .base import InpaintModel
from .schema import Config

class Flux(InpaintModel):
    """Flux inpainting using the Diffusers pipeline."""

    name = "flux"
    pad_mod = 32

    def init_model(self, device, **kwargs):
        model_name = os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.1-schnell")
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.pipe = FluxInpaintPipeline.from_pretrained(model_name, torch_dtype=dtype)
        self.pipe = self.pipe.to(device)
        self.device = device

    @staticmethod
    def is_downloaded() -> bool:
        return True

    def forward(self, image: np.ndarray, mask: np.ndarray, config: Config):
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask)
        result = self.pipe(
            prompt="",
            image=image_pil,
            mask_image=mask_pil,
            height=image.shape[0],
            width=image.shape[1],
            strength=1.0,
            num_inference_steps=20,
        ).images[0]
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
