import argparse
import io
import json
import re
import gc
from pathlib import Path
from typing import Optional

import fitz
import gradio as gr
import numpy as np
import rarfile
import torch
from PIL import Image
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class ComicAnimator:
    def __init__(
        self,
        vlm_model: str,
        video_model: str,
        comics_dir: str,
        device: str = "cuda"
    ):
        self.device = device
        self.comics_dir = Path(comics_dir)
        self.pages: list[Image.Image] = []
        self.current_page_idx: int = 0
        self.output_dir = Path("./outputs")
        self.output_dir.mkdir(exist_ok=True)

        print("Loading VLM for motion analysis...")
        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            vlm_model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_model)

        print("Loading CogVideoX for animation...")
        self.video_pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            video_model,
            torch_dtype=torch.bfloat16
        )
        self.video_pipe.to(device)
        self.video_pipe.enable_sequential_cpu_offload()
        self.video_pipe.vae.enable_slicing()
        self.video_pipe.vae.enable_tiling()

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

    def _vlm_query(self, image: Image.Image, prompt: str, max_tokens: int = 400) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
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
            output_ids = self.vlm.generate(**inputs, max_new_tokens=max_tokens, temperature=0.3)

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def analyze_panel_for_animation(self, panel_image: Image.Image) -> dict:
        prompt = """You are an animation director for a comic book movie. Analyze this comic panel and describe EXACTLY what motion/animation should happen.

Look at:
1. CHARACTERS: What are they doing? What body parts should move?
2. MECHANICAL PARTS: Any robots, tentacles, weapons - how should they move?
3. EFFECTS: Speed lines, impact bursts, energy - what motion do they suggest?
4. EMOTIONS: Character expressions - should they change?

Create a DETAILED motion description for a 3-second animation. Be SPECIFIC about:
- Starting position → Ending position
- Direction of movement (left, right, up, down, rotating)
- Speed (slow, medium, fast)
- What stays still vs what moves

Return a single paragraph prompt that describes ALL the motion in detail. This will be used directly by a video AI model.

Example good prompt: "The muscular man raises both fists triumphantly while his four mechanical tentacle arms wave and undulate menacingly around him. The tentacles move in a serpentine motion, their claw-like ends opening and closing. The man's expression shifts from neutral to a confident smile. Slight camera zoom in on the figure."

Now analyze this panel and write a similar motion prompt:"""

        response = self._vlm_query(panel_image, prompt, max_tokens=300)
        print(f"Motion analysis: {response}")

        clean_prompt = response.strip()
        if len(clean_prompt) < 20:
            clean_prompt = "Subtle animation with gentle movement, slight camera motion, and atmospheric effects."

        return {"motion_prompt": clean_prompt}

    def animate_panel(
        self,
        panel_image: Image.Image,
        motion_prompt: str,
        num_frames: int = 49,
        fps: int = 8,
        progress_callback=None
    ) -> str:
        target_size = (720, 480)
        
        img_w, img_h = panel_image.size
        aspect = img_w / img_h
        target_aspect = target_size[0] / target_size[1]

        if aspect > target_aspect:
            new_w = target_size[0]
            new_h = int(new_w / aspect)
        else:
            new_h = target_size[1]
            new_w = int(new_h * aspect)

        new_w = (new_w // 16) * 16
        new_h = (new_h // 16) * 16
        new_w = max(new_w, 256)
        new_h = max(new_h, 256)

        resized = panel_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        padded = Image.new("RGB", target_size, (0, 0, 0))
        paste_x = (target_size[0] - new_w) // 2
        paste_y = (target_size[1] - new_h) // 2
        padded.paste(resized, (paste_x, paste_y))

        full_prompt = f"Comic book animation, maintain illustrated art style: {motion_prompt}"

        print(f"Generating video with prompt: {full_prompt[:100]}...")

        generator = torch.Generator(device=self.device).manual_seed(42)

        with torch.no_grad():
            video_frames = self.video_pipe(
                image=padded,
                prompt=full_prompt,
                num_frames=num_frames,
                num_inference_steps=30,
                guidance_scale=6.0,
                generator=generator
            ).frames[0]

        output_path = self.output_dir / f"comic_animation_{np.random.randint(10000, 99999)}.mp4"
        export_to_video(video_frames, str(output_path), fps=fps)

        gc.collect()
        torch.cuda.empty_cache()

        print(f"Video saved: {output_path}")
        return str(output_path)

    def process_click(
        self,
        image: Optional[np.ndarray],
        evt: gr.SelectData,
        progress=gr.Progress()
    ) -> tuple[str, str, Optional[str]]:
        if image is None or len(self.pages) == 0:
            return "No image loaded", "", None

        click_x, click_y = evt.index
        pil_image = self.pages[self.current_page_idx]
        img_w, img_h = pil_image.size

        region_size = min(img_w, img_h) // 2
        x1 = max(0, click_x - region_size // 2)
        y1 = max(0, click_y - region_size // 2)
        x2 = min(img_w, x1 + region_size)
        y2 = min(img_h, y1 + region_size)

        panel = pil_image.crop((x1, y1, x2, y2))

        print(f"\n{'='*50}")
        print(f"Animating panel [{x1},{y1}] to [{x2},{y2}]")
        print(f"{'='*50}")

        progress(0.1, desc="Analyzing panel for motion...")
        motion_data = self.analyze_panel_for_animation(panel)
        motion_prompt = motion_data["motion_prompt"]

        progress(0.2, desc="Generating animation (~60-90 seconds)...")
        video_path = self.animate_panel(panel, motion_prompt)

        description = f"""### Animation Generated

**Motion Prompt:**
{motion_prompt}

**Output:** {video_path}
"""
        progress(1.0, desc="Done!")
        return description, motion_prompt, video_path


def build_ui(animator: ComicAnimator) -> gr.Blocks:
    with gr.Blocks(title="AI Comic Animator", theme=gr.themes.Soft()) as app:
        gr.Markdown("""# 🎬 AI Comic Animator (CogVideoX)

Load a comic and click on any panel to generate a ~3-5 second animation.
The AI analyzes the scene and creates motion while preserving the comic art style.""")

        with gr.Row():
            comic_dropdown = gr.Dropdown(
                choices=animator.list_comics(),
                label="Select Comic",
                interactive=True
            )
            load_btn = gr.Button("📖 Load Comic", variant="primary")
            refresh_btn = gr.Button("🔄 Refresh")

        with gr.Row():
            page_slider = gr.Slider(0, 0, step=1, label="Page", interactive=True)

        with gr.Row():
            with gr.Column(scale=1):
                comic_image = gr.Image(label="📖 Comic Page - Click to animate a panel")
            with gr.Column(scale=1):
                video_output = gr.Video(label="🎬 Generated Animation", autoplay=True)

        with gr.Row():
            motion_prompt_display = gr.Textbox(label="Motion Prompt Used", lines=3, interactive=False)

        with gr.Row():
            description_output = gr.Markdown("*Click a panel to generate animation*")

        def refresh_list():
            return gr.update(choices=animator.list_comics())

        def load_comic(filename):
            if not filename:
                return None, gr.update(maximum=0, value=0)
            animator.load_comic(filename)
            if len(animator.pages) == 0:
                return None, gr.update(maximum=0, value=0)
            max_page = len(animator.pages) - 1
            return np.array(animator.pages[0]), gr.update(maximum=max_page, value=0)

        def change_page(page_num):
            if len(animator.pages) == 0:
                return None
            animator.current_page_idx = int(page_num)
            return np.array(animator.pages[animator.current_page_idx])

        refresh_btn.click(refresh_list, outputs=[comic_dropdown])
        load_btn.click(load_comic, inputs=[comic_dropdown], outputs=[comic_image, page_slider])
        page_slider.change(change_page, inputs=[page_slider], outputs=[comic_image])
        comic_image.select(
            animator.process_click,
            inputs=[comic_image],
            outputs=[description_output, motion_prompt_display, video_output]
        )

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--video-model", default="THUDM/CogVideoX-5b-I2V")
    parser.add_argument("--comics-dir", default="./comics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7777)
    args = parser.parse_args()

    animator = ComicAnimator(
        vlm_model=args.vlm_model,
        video_model=args.video_model,
        comics_dir=args.comics_dir,
        device=args.device
    )

    app = build_ui(animator)
    app.launch(server_name=args.host, server_port=args.port, share=True)


if __name__ == "__main__":
    main()