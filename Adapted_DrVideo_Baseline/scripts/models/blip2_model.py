# models/blip2_model.py
# BLIP base (Salesforce/blip-image-captioning-base) captioner for CPU or CUDA.
# Compatible with torch==1.9.x and transformers==4.31.0.
# Keep it lightweight and robust for baseline.

from typing import Any
import torch
from PIL import Image
import numpy as np
# robust imports for BLIP on recent Transformers
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except Exception:
    from transformers.models.blip.processing_blip import BlipProcessor
    from transformers.models.blip.modeling_blip import BlipForConditionalGeneration



class ImageCaptioner:
    def __init__(self, model_name: str = "blip", device: str = "cpu"):
        """
        model_name: kept for CLI compatibility; only 'blip' is implemented here.
        device: 'cpu' or 'cuda' (use 'cpu' on mac).
        """
        self.model_name = model_name or "blip"
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.processor, self.model = self._initialize_model()

    def _initialize_model(self):
        # BLIP base is ~0.9GB, runs on CPU (slower but acceptable for sparse frames).
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model.to(self.device)
        # Do NOT call model.half(); half-precision is not supported on CPU.
        return processor, model

    def image_caption(self, image_rgb: np.ndarray, prompt: str, type_info: str) -> str:
        """
        image_rgb: HxWxC RGB numpy array
        prompt: guiding text prompt; BLIP supports conditional captioning with 'text=...'
        type_info: 'A'|'B'|'C' -> controls max_new_tokens only
        """
        image_pil = Image.fromarray(image_rgb)
        inputs = self.processor(images=image_pil, text=prompt, return_tensors="pt")
        # move tensors to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        max_new = 50 if type_info in ("A", "B") else 100
        # greedy decoding is fine for baseline
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new)
        text = self.processor.decode(out_ids[0], skip_special_tokens=True).strip()

        # BLIP often echoes the prompt; strip it if present
        if text.startswith(prompt):
            text = text[len(prompt):].strip(" :.-")
        return text

    def image_caption_debug(self, image_src: Any) -> str:
        return "A person near a screen with slides."
