import argparse
import io
import gc
from pathlib import Path
from typing import Optional

import fitz
import gradio as gr
import numpy as np
import rarfile
import torch
from PIL import Image
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, CLIPVisionModel
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

        print("Loading Wan2.1 I2V pipeline...")
        
        image_encoder = CLIPVisionModel.from_pretrained(
            video_model, 
            subfolder="image_encoder", 
            torch_dtype=torch.float32
        )
        
        vae = AutoencoderKLWan.from_pretrained(
            video_model, 
            subfolder="vae", 
            torch_dtype=torch.float32
        )
        
        self.video_pipe = WanImageToVideoPipeline.from_pretrained(
            video_model,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=torch.bfloat16
        )
        self.video_pipe.to(device)

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
            output_ids = self.vlm.generate(**inputs, max_new_tokens=max_tokens, temperature=0.2)

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def analyze_panel_for_animation(self, panel_image: Image.Image) -> str:
        prompt = """Analyze this comic book panel and create a detailed motion description for animation.

Describe what should move and how:
- Character body movements (arms raising, turning, walking)
- Mechanical parts (tentacles waving, robots moving, weapons swinging)
- Effects (energy blasts, speed lines becoming animated)
- Facial expressions changing
- Camera movement (zoom, pan)

Write ONE detailed paragraph describing the animation. Be specific about directions and speeds.

Example: "The villain raises his mechanical tentacle arms in a triumphant pose, the four metallic appendages undulating and coiling like serpents. His human arms pump upward with fists clenched. The tentacle claws open and close menacingly. His expression shifts to a sinister grin. Slight camera push in toward the figure."

Write the animation description for this panel:"""

        response = self._vlm_query(panel_image, prompt, max_tokens=200)
        print(f"VLM Motion Analysis: {response}")

        motion_prompt = response.strip().replace('"', '').replace('\n', ' ')
        
        if len(motion_prompt) < 20:
            motion_prompt = "Subtle character movement with gentle animation, maintaining comic book art style"

        full_prompt = f"Comic book panel animation, illustrated art style, hand-drawn aesthetic: {motion_prompt}"
        
        return full_prompt

    def prepare_image_for_wan(self, panel_image: Image.Image) -> Image.Image:
        target_h = 480
        target_w = 832
        
        img_w, img_h = panel_image.size
        
        scale = max(target_w / img_w, target_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        resized = panel_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        cropped = resized.crop((left, top, left + target_w, top + target_h))
        
        return cropped

    def animate_panel(
        self,
        panel_image: Image.Image,
        motion_prompt: str,
    ) -> str:
        prepared_image = self.prepare_image_for_wan(panel_image)
        
        print(f"Prepared image size: {prepared_image.size}")
        print(f"Motion prompt: {motion_prompt[:100]}...")

        negative_prompt = (
            "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
            "realistic photo, 3D render, style change, watermark, text corruption, "
            "extra limbs, missing limbs, floating objects"
        )

        generator = torch.Generator(device=self.device).manual_seed(42)

        with torch.no_grad():
            output = self.video_pipe(
                image=prepared_image,
                prompt=motion_prompt,
                negative_prompt=negative_prompt,
                height=480,
                width=832,
                num_frames=81,
                num_inference_steps=30,
                guidance_scale=5.0,
                generator=generator
            )
            video_frames = output.frames[0]

        output_path = self.output_dir / f"comic_anim_{np.random.randint(10000, 99999)}.mp4"
        export_to_video(video_frames, str(output_path), fps=16)

        gc.collect()
        torch.cuda.empty_cache()

        print(f"Video saved: {output_path}")
        return str(output_path)

    def process_click(
        self,
        image: Optional[np.ndarray],
        evt: gr.SelectData,
        progress=gr.Progress()
    ) -> tuple[str, str, Optional[str], Optional[Image.Image]]:
        if image is None or len(self.pages) == 0:
            return "No image loaded", "", None, None

        click_x, click_y = evt.index
        pil_image = self.pages[self.current_page_idx]
        img_w, img_h = pil_image.size

        panel_size = min(img_w, img_h) * 2 // 3
        
        x1 = max(0, click_x - panel_size // 2)
        y1 = max(0, click_y - panel_size // 2)
        x2 = min(img_w, x1 + panel_size)
        y2 = min(img_h, y1 + panel_size)
        
        if x2 - x1 < panel_size:
            x1 = max(0, x2 - panel_size)
        if y2 - y1 < panel_size:
            y1 = max(0, y2 - panel_size)

        panel = pil_image.crop((x1, y1, x2, y2))

        print(f"\n{'='*60}")
        print(f"Selected panel: [{x1},{y1}] to [{x2},{y2}]")
        print(f"Panel size: {panel.size}")
        print(f"{'='*60}")

        progress(0.1, desc="Analyzing panel for motion...")
        motion_prompt = self.analyze_panel_for_animation(panel)

        progress(0.2, desc="Generating animation with Wan2.1 (60-120s)...")
        video_path = self.animate_panel(panel, motion_prompt)

        description = f"""### Animation Complete

**Selected Region:** [{x1},{y1}] → [{x2},{y2}]

**Motion Prompt:**
{motion_prompt}
"""
        progress(1.0, desc="Done!")
        return description, motion_prompt, video_path, panel


def build_ui(animator: ComicAnimator) -> gr.Blocks:
    with gr.Blocks(title="AI Comic Animator - Wan2.1", theme=gr.themes.Soft()) as app:
        gr.Markdown("""# 🎬 AI Comic Animator (Wan2.1)

Click on any panel to animate it with AI. The system will:
1. Extract the clicked panel region
2. Analyze the scene to determine motion
3. Generate a ~5 second animation preserving comic art style""")

        with gr.Row():
            comic_dropdown = gr.Dropdown(
                choices=animator.list_comics(),
                label="Comic File",
                interactive=True
            )
            load_btn = gr.Button("📖 Load", variant="primary")
            refresh_btn = gr.Button("🔄")

        with gr.Row():
            page_slider = gr.Slider(0, 0, step=1, label="Page", interactive=True)

        with gr.Row():
            with gr.Column(scale=1):
                comic_image = gr.Image(label="📖 Click to select panel")
            with gr.Column(scale=1):
                selected_panel = gr.Image(label="Selected Panel", height=250)
                video_output = gr.Video(label="🎬 Animation", autoplay=True, height=300)

        with gr.Row():
            motion_prompt_box = gr.Textbox(label="Motion Prompt", lines=3, interactive=False)

        with gr.Row():
            status_output = gr.Markdown("*Click a panel to animate*")

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
            outputs=[status_output, motion_prompt_box, video_output, selected_panel]
        )

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--video-model", default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
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