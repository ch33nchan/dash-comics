import argparse
import io
import os
from pathlib import Path
from typing import Optional

import fitz
import gradio as gr
import numpy as np
import rarfile
import torch
from PIL import Image
from diffusers import AudioLDMPipeline
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class ComicReader:
    def __init__(self, vlm_model: str, audio_model: str, comics_dir: str, device: str = "cuda"):
        self.device = device
        self.comics_dir = Path(comics_dir)
        self.pages: list[Image.Image] = []
        self.current_page_idx: int = 0
        
        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            vlm_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_model)
        
        self.audio_pipe = AudioLDMPipeline.from_pretrained(
            audio_model,
            torch_dtype=torch.float16
        ).to(device)

    def list_comics(self) -> list[str]:
        if not self.comics_dir.exists():
            self.comics_dir.mkdir(parents=True, exist_ok=True)
            return []
        cbr_files = list(self.comics_dir.glob("*.cbr"))
        pdf_files = list(self.comics_dir.glob("*.pdf"))
        return sorted([f.name for f in cbr_files + pdf_files])

    def load_comic(self, filename: str) -> list[Image.Image]:
        if not filename:
            return []
        self.pages = []
        filepath = self.comics_dir / filename
        
        if filename.lower().endswith(".cbr"):
            self._load_cbr(filepath)
        elif filename.lower().endswith(".pdf"):
            self._load_pdf(filepath)
        
        self.current_page_idx = 0
        return self.pages

    def _load_cbr(self, cbr_path: Path) -> None:
        with rarfile.RarFile(cbr_path) as rf:
            image_files = sorted([
                f for f in rf.namelist()
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
            ])
            for img_file in image_files:
                data = rf.read(img_file)
                img = Image.open(io.BytesIO(data)).convert("RGB")
                self.pages.append(img)

    def _load_pdf(self, pdf_path: Path, dpi: int = 150) -> None:
        doc = fitz.open(pdf_path)
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            self.pages.append(img)
        doc.close()

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

    def generate_sound(self, description: str, duration: float = 2.0) -> tuple[int, np.ndarray]:
        prompt = f"comic book sound effect: {description}"
        
        generator = torch.Generator(device=self.device).manual_seed(np.random.randint(0, 2**32))
        
        with torch.no_grad():
            audio = self.audio_pipe(
                prompt,
                num_inference_steps=20,
                audio_length_in_s=duration,
                generator=generator
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
        
        click_x, click_y = evt.index
        pil_image = self.pages[self.current_page_idx]
        img_w, img_h = pil_image.size
        
        region_size = min(img_w, img_h) // 4
        
        x1 = max(0, click_x - region_size // 2)
        y1 = max(0, click_y - region_size // 2)
        x2 = min(img_w, x1 + region_size)
        y2 = min(img_h, y1 + region_size)
        
        description = self.analyze_region(pil_image, x1, y1, x2 - x1, y2 - y1)
        audio = self.generate_sound(description)
        
        return f"[{x1},{y1} -> {x2},{y2}] {description}", audio


def build_ui(reader: ComicReader) -> gr.Blocks:
    with gr.Blocks(title="AI Comic Reader") as app:
        gr.Markdown("# AI Comic Reader\nSelect a comic (CBR/PDF), click on panels to hear them come alive.")
        
        with gr.Row():
            comic_dropdown = gr.Dropdown(
                choices=reader.list_comics(),
                label="Select Comic",
                interactive=True
            )
            load_btn = gr.Button("Load")
            refresh_btn = gr.Button("Refresh")
            reset_btn = gr.Button("Reset")
            page_slider = gr.Slider(0, 0, step=1, label="Page", interactive=True)
        
        with gr.Row():
            comic_image = gr.Image(label="Comic Page", interactive=True)
        
        with gr.Row():
            description_output = gr.Textbox(label="Scene Description")
            audio_output = gr.Audio(label="Sound Effect")

        def refresh_list():
            return gr.update(choices=reader.list_comics())

        def load_comic(filename):
            if not filename:
                return None, gr.update(maximum=0, value=0)
            reader.load_comic(filename)
            if len(reader.pages) == 0:
                return None, gr.update(maximum=0, value=0)
            max_page = len(reader.pages) - 1
            return np.array(reader.pages[0]), gr.update(maximum=max_page, value=0)

        def change_page(page_num):
            if len(reader.pages) == 0:
                return None
            reader.current_page_idx = int(page_num)
            return np.array(reader.pages[reader.current_page_idx])

        def reset_all():
            reader.pages = []
            reader.current_page_idx = 0
            return (
                gr.update(value=None),
                gr.update(maximum=0, value=0),
                "",
                None,
                gr.update(value=None)
            )

        refresh_btn.click(refresh_list, outputs=[comic_dropdown])
        load_btn.click(load_comic, inputs=[comic_dropdown], outputs=[comic_image, page_slider])
        comic_dropdown.change(load_comic, inputs=[comic_dropdown], outputs=[comic_image, page_slider])
        page_slider.change(change_page, inputs=[page_slider], outputs=[comic_image])
        comic_image.select(reader.process_click, inputs=[comic_image], outputs=[description_output, audio_output])
        reset_btn.click(reset_all, outputs=[comic_dropdown, page_slider, description_output, audio_output, comic_image])

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--audio-model", default="cvssp/audioldm-s-full-v2")
    parser.add_argument("--comics-dir", default="./comics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7777)
    args = parser.parse_args()

    reader = ComicReader(
        vlm_model=args.vlm_model,
        audio_model=args.audio_model,
        comics_dir=args.comics_dir,
        device=args.device
    )
    
    app = build_ui(reader)
    app.launch(server_name=args.host, server_port=args.port, share=True)


if __name__ == "__main__":
    main()