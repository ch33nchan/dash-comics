import argparse
import io
import os
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import rarfile
import torch
from PIL import Image
from diffusers import AudioLDM2Pipeline
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import scipy.io.wavfile as wavfile


class ComicReader:
    def __init__(self, vlm_model: str, audio_model: str, device: str = "cuda"):
        self.device = device
        self.pages: list[Image.Image] = []
        self.current_page_idx: int = 0
        
        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            vlm_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_model)
        
        self.audio_pipe = AudioLDM2Pipeline.from_pretrained(
            audio_model,
            torch_dtype=torch.float16
        ).to(device)

    def load_cbr(self, cbr_path: str) -> list[Image.Image]:
        self.pages = []
        with rarfile.RarFile(cbr_path) as rf:
            image_files = sorted([
                f for f in rf.namelist()
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
            ])
            for img_file in image_files:
                data = rf.read(img_file)
                img = Image.open(io.BytesIO(data)).convert("RGB")
                self.pages.append(img)
        self.current_page_idx = 0
        return self.pages

    def analyze_region(self, image: Image.Image, x: int, y: int, w: int, h: int) -> str:
        region = image.crop((x, y, x + w, y + h))
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": region},
                    {"type": "text", "text": (
                        "Describe the action in this comic panel in one sentence. "
                        "Focus on: character actions, motion, impact sounds, emotions. "
                        "If there's onomatopoeia text like BAM, WHOOSH, POW, include it."
                    )}
                ]
            }
        ]
        
        text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.vlm.generate(**inputs, max_new_tokens=128)
        
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        description = self.vlm_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return description

    def generate_sound(self, description: str, duration: float = 3.0) -> tuple[int, np.ndarray]:
        prompt = f"comic book sound effect: {description}"
        
        with torch.no_grad():
            audio = self.audio_pipe(
                prompt,
                num_inference_steps=50,
                audio_length_in_s=duration
            ).audios[0]
        
        sample_rate = 16000
        audio_int16 = (audio * 32767).astype(np.int16)
        return sample_rate, audio_int16

    def process_click(
        self, 
        image: Optional[np.ndarray], 
        evt: gr.SelectData
    ) -> tuple[str, tuple[int, np.ndarray]]:
        if image is None or len(self.pages) == 0:
            return "No image loaded", (16000, np.zeros(16000, dtype=np.int16))
        
        x, y = evt.index
        h, w = image.shape[:2]
        region_size = min(w, h) // 3
        
        x1 = max(0, x - region_size // 2)
        y1 = max(0, y - region_size // 2)
        
        pil_image = self.pages[self.current_page_idx]
        description = self.analyze_region(pil_image, x1, y1, region_size, region_size)
        audio = self.generate_sound(description)
        
        return description, audio


def build_ui(reader: ComicReader) -> gr.Blocks:
    with gr.Blocks(title="AI Comic Reader") as app:
        gr.Markdown("# AI Comic Reader\nLoad a CBR file, click on panels to hear them come alive.")
        
        with gr.Row():
            cbr_input = gr.File(label="Upload CBR", file_types=[".cbr"])
            page_slider = gr.Slider(0, 0, step=1, label="Page", interactive=True)
        
        with gr.Row():
            comic_image = gr.Image(label="Comic Page", interactive=True)
        
        with gr.Row():
            description_output = gr.Textbox(label="Scene Description")
            audio_output = gr.Audio(label="Sound Effect")

        def load_comic(file):
            if file is None:
                return None, gr.update(maximum=0, value=0)
            reader.load_cbr(file.name)
            max_page = len(reader.pages) - 1
            return np.array(reader.pages[0]), gr.update(maximum=max_page, value=0)

        def change_page(page_num):
            reader.current_page_idx = int(page_num)
            return np.array(reader.pages[reader.current_page_idx])

        cbr_input.change(load_comic, inputs=[cbr_input], outputs=[comic_image, page_slider])
        page_slider.change(change_page, inputs=[page_slider], outputs=[comic_image])
        comic_image.select(reader.process_click, inputs=[comic_image], outputs=[description_output, audio_output])

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--audio-model", default="cvssp/audioldm2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    reader = ComicReader(
        vlm_model=args.vlm_model,
        audio_model=args.audio_model,
        device=args.device
    )
    
    app = build_ui(reader)
    app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()