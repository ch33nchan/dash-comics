import argparse
import io
import json
import re
import tempfile
from pathlib import Path
from typing import Optional

import fitz
import gradio as gr
import numpy as np
import rarfile
import torch
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
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

        print("Loading VLM...")
        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            vlm_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_model)

        print("Loading Stable Video Diffusion...")
        self.video_pipe = StableVideoDiffusionPipeline.from_pretrained(
            video_model,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        self.video_pipe.to(device)
        self.video_pipe.enable_model_cpu_offload()

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

    def _vlm_query(self, images: list[Image.Image], prompt: str, max_tokens: int = 500) -> str:
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

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

    def _get_context_panels(self, page_img: Image.Image, x: int, y: int, w: int, h: int) -> dict:
        current_panel = page_img.crop((x, y, x + w, y + h))
        
        img_w, img_h = page_img.size
        
        prev_panel = None
        next_panel = None
        
        if y > h:
            prev_y = max(0, y - h - 20)
            prev_panel = page_img.crop((x, prev_y, x + w, prev_y + h))
        elif x > w:
            prev_x = max(0, x - w - 20)
            prev_panel = page_img.crop((prev_x, y, prev_x + w, y + h))
        
        if y + h * 2 < img_h:
            next_y = min(img_h - h, y + h + 20)
            next_panel = page_img.crop((x, next_y, x + w, next_y + h))
        elif x + w * 2 < img_w:
            next_x = min(img_w - w, x + w + 20)
            next_panel = page_img.crop((next_x, y, next_x + w, y + h))

        return {
            "current": current_panel,
            "previous": prev_panel,
            "next": next_panel
        }

    def analyze_for_animation(self, panels: dict) -> dict:
        images = [panels["current"]]
        context_desc = ""
        
        if panels["previous"]:
            images.insert(0, panels["previous"])
            context_desc += "First image: PREVIOUS panel. "
        if panels["next"]:
            images.append(panels["next"])
            context_desc += "Last image: NEXT panel. "
        
        if len(images) == 1:
            context_desc = "Single panel to animate."

        prompt = f"""You are an animation director. Analyze these comic panel(s) to plan a 3-second animation.

{context_desc}
The MIDDLE/MAIN image is the panel to animate.

Describe EXACTLY what motion should happen in a 3-second animation:

1. CHARACTERS: List each character/figure visible and their motion
   - Starting pose → Ending pose
   - Body movement (walking, jumping, punching, turning)
   - Facial expression changes
   - Hand/arm gestures

2. OBJECTS: Any objects that should move
   - Projectiles, weapons, vehicles
   - Environmental elements (doors, papers, debris)

3. EFFECTS: Visual effects to animate
   - Speed lines, impact bursts
   - Explosions, smoke, fire
   - Text/onomatopoeia (should pulse or shake)

4. CAMERA: Suggested camera motion
   - Zoom in/out
   - Pan left/right/up/down
   - Shake for impact

5. MOTION_INTENSITY: Rate 1-10 how much motion (1=subtle, 10=intense action)

Based on story context (what happened before, what happens after), the animation should feel continuous.

Return ONLY this JSON:
{{
  "scene_description": "Brief description of what's happening",
  "characters": [
    {{"name": "Hero", "motion": "Throws punch from left to right, body rotating forward"}}
  ],
  "objects": [
    {{"name": "Debris", "motion": "Fragments fly outward from impact point"}}
  ],
  "effects": [
    {{"type": "impact_burst", "motion": "Expands rapidly then fades"}}
  ],
  "camera": "Slight zoom in with shake on impact",
  "motion_intensity": 7,
  "motion_prompt": "A single clear sentence describing the main motion for the video model"
}}"""

        response = self._vlm_query(images, prompt, max_tokens=600)
        print(f"Animation analysis: {response}")

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        return {
            "scene_description": "Comic panel with action",
            "characters": [],
            "objects": [],
            "effects": [],
            "camera": "slight zoom",
            "motion_intensity": 5,
            "motion_prompt": "Subtle movement and camera pan across the scene"
        }

    def animate_panel(
        self,
        panel_image: Image.Image,
        motion_data: dict,
        num_frames: int = 25,
        fps: int = 8
    ) -> str:
        target_size = (1024, 576)
        
        img_w, img_h = panel_image.size
        aspect = img_w / img_h
        target_aspect = target_size[0] / target_size[1]
        
        if aspect > target_aspect:
            new_w = target_size[0]
            new_h = int(new_w / aspect)
        else:
            new_h = target_size[1]
            new_w = int(new_h * aspect)
        
        resized = panel_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        padded = Image.new("RGB", target_size, (0, 0, 0))
        paste_x = (target_size[0] - new_w) // 2
        paste_y = (target_size[1] - new_h) // 2
        padded.paste(resized, (paste_x, paste_y))

        motion_intensity = motion_data.get("motion_intensity", 5)
        motion_bucket_id = min(255, max(1, motion_intensity * 25))

        print(f"Generating video with motion_bucket_id={motion_bucket_id}...")

        generator = torch.Generator(device=self.device).manual_seed(np.random.randint(0, 2**32))

        with torch.no_grad():
            frames = self.video_pipe(
                padded,
                num_frames=num_frames,
                decode_chunk_size=8,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=0.1,
                generator=generator
            ).frames[0]

        output_path = self.output_dir / f"animation_{np.random.randint(10000, 99999)}.mp4"
        export_to_video(frames, str(output_path), fps=fps)

        print(f"Video saved to {output_path}")
        return str(output_path)

    def process_click(
        self,
        image: Optional[np.ndarray],
        evt: gr.SelectData,
        progress=gr.Progress()
    ) -> tuple[str, Optional[str]]:
        if image is None or len(self.pages) == 0:
            return "No image loaded", None

        click_x, click_y = evt.index
        pil_image = self.pages[self.current_page_idx]
        img_w, img_h = pil_image.size

        region_size = min(img_w, img_h) // 3
        x1 = max(0, click_x - region_size // 2)
        y1 = max(0, click_y - region_size // 2)
        x2 = min(img_w, x1 + region_size)
        y2 = min(img_h, y1 + region_size)

        print(f"\n{'='*50}")
        print(f"Animating panel at [{x1},{y1}] to [{x2},{y2}]")
        print(f"{'='*50}")

        progress(0.1, desc="Getting context panels...")
        panels = self._get_context_panels(pil_image, x1, y1, x2 - x1, y2 - y1)

        progress(0.2, desc="Analyzing scene for animation...")
        motion_data = self.analyze_for_animation(panels)

        progress(0.4, desc="Generating animation (this takes ~30-60s)...")
        video_path = self.animate_panel(panels["current"], motion_data)

        description = f"""### Animation Plan

**Scene:** {motion_data.get('scene_description', 'N/A')}

**Characters:**
"""
        for char in motion_data.get('characters', []):
            description += f"- **{char.get('name', '?')}**: {char.get('motion', 'N/A')}\n"

        description += "\n**Objects:**\n"
        for obj in motion_data.get('objects', []):
            description += f"- **{obj.get('name', '?')}**: {obj.get('motion', 'N/A')}\n"

        description += "\n**Effects:**\n"
        for fx in motion_data.get('effects', []):
            description += f"- **{fx.get('type', '?')}**: {fx.get('motion', 'N/A')}\n"

        description += f"""
**Camera:** {motion_data.get('camera', 'N/A')}

**Motion Intensity:** {motion_data.get('motion_intensity', 'N/A')}/10

**Motion Prompt:** {motion_data.get('motion_prompt', 'N/A')}
"""

        progress(1.0, desc="Done!")
        return description, video_path


def build_ui(animator: ComicAnimator) -> gr.Blocks:
    with gr.Blocks(title="AI Comic Animator", theme=gr.themes.Soft()) as app:
        gr.Markdown("""# 🎬 AI Comic Animator

Load a comic, click on any panel to bring it to life with AI-generated animation.
The animation considers context from surrounding panels for continuity.""")

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
                comic_image = gr.Image(label="📖 Comic Page - Click a panel to animate")
            with gr.Column(scale=1):
                video_output = gr.Video(label="🎬 Animation", autoplay=True)

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
            outputs=[description_output, video_output]
        )

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--video-model", default="stabilityai/stable-video-diffusion-img2vid-xt")
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