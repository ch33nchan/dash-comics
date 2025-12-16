import argparse
import io
import json
import re
from pathlib import Path
from typing import Optional

import fitz
import gradio as gr
import numpy as np
import rarfile
import torch
import torchaudio
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BarkModel, AutoTokenizer
from qwen_vl_utils import process_vision_info
from diffusers import AudioLDMPipeline


BARK_VOICES = [
    "v2/en_speaker_0",
    "v2/en_speaker_1",
    "v2/en_speaker_2",
    "v2/en_speaker_3",
    "v2/en_speaker_4",
    "v2/en_speaker_5",
    "v2/en_speaker_6",
    "v2/en_speaker_7",
    "v2/en_speaker_8",
    "v2/en_speaker_9",
]


class ComicReader:
    def __init__(
        self,
        vlm_model: str,
        tts_model: str,
        sfx_model: str,
        comics_dir: str,
        device: str = "cuda"
    ):
        self.device = device
        self.comics_dir = Path(comics_dir)
        self.pages: list[Image.Image] = []
        self.current_page_idx: int = 0
        self.characters: dict[str, str] = {}

        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            vlm_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_model)

        self.tts = BarkModel.from_pretrained(tts_model, torch_dtype=torch.float16).to(device)
        self.tts_tokenizer = AutoTokenizer.from_pretrained(tts_model)
        self.tts_sample_rate = 24000

        self.sfx_pipe = AudioLDMPipeline.from_pretrained(
            sfx_model,
            torch_dtype=torch.float16
        ).to(device)
        self.sfx_sample_rate = 16000

    def list_comics(self) -> list[str]:
        if not self.comics_dir.exists():
            self.comics_dir.mkdir(parents=True, exist_ok=True)
            return []
        cbr_files = list(self.comics_dir.glob("*.cbr"))
        pdf_files = list(self.comics_dir.glob("*.pdf"))
        return sorted([f.name for f in cbr_files + pdf_files])

    def load_comic(self, filename: str, progress=gr.Progress()) -> tuple[list[Image.Image], str]:
        if not filename:
            return [], ""
        self.pages = []
        self.characters = {}
        filepath = self.comics_dir / filename

        if filename.lower().endswith(".cbr"):
            self._load_cbr(filepath)
        elif filename.lower().endswith(".pdf"):
            self._load_pdf(filepath)

        self.current_page_idx = 0

        progress(0, desc="Analyzing characters...")
        char_report = self._analyze_characters(progress)

        return self.pages, char_report

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

    def _analyze_characters(self, progress) -> str:
        if not self.pages:
            return "No pages loaded"

        all_characters = set()
        sample_pages = self.pages[:min(5, len(self.pages))]

        for i, page in enumerate(sample_pages):
            progress((i + 1) / len(sample_pages), desc=f"Scanning page {i + 1}/{len(sample_pages)}...")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": page},
                        {"type": "text", "text": """List ALL unique character names visible in this comic page.
Look for:
- Names in speech bubbles
- Character labels
- Names mentioned in dialogue
- Visual character identification

Respond ONLY with a JSON array of character names:
["Character1", "Character2", "Character3"]

If no names found, analyze visual appearances and give descriptive names like:
["Hero in red suit", "Villain with mask", "Old man"]"""}
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
                output_ids = self.vlm.generate(**inputs, max_new_tokens=200, temperature=0.1)

            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            response = self.vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            try:
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    chars = json.loads(json_match.group())
                    all_characters.update(chars)
            except json.JSONDecodeError:
                pass

        char_list = list(all_characters)
        for i, char in enumerate(char_list):
            voice_idx = i % len(BARK_VOICES)
            self.characters[char.lower()] = BARK_VOICES[voice_idx]

        report = "**Detected Characters & Voice Assignments:**\n"
        for char, voice in self.characters.items():
            report += f"- {char.title()}: {voice}\n"

        if not self.characters:
            self.characters["narrator"] = BARK_VOICES[0]
            self.characters["unknown"] = BARK_VOICES[1]
            report = "No named characters detected. Using default voices."

        return report

    def _get_voice_for_character(self, speaker: str) -> str:
        speaker_lower = speaker.lower()
        if speaker_lower in self.characters:
            return self.characters[speaker_lower]
        for char_name, voice in self.characters.items():
            if char_name in speaker_lower or speaker_lower in char_name:
                return voice
        if not self.characters:
            return BARK_VOICES[0]
        return list(self.characters.values())[hash(speaker) % len(self.characters)]

    def analyze_panel(self, image: Image.Image, x: int, y: int, w: int, h: int) -> dict:
        region = image.crop((x, y, x + w, y + h))

        char_list = list(self.characters.keys()) if self.characters else ["unknown speaker"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": region},
                    {"type": "text", "text": f"""Analyze this comic panel. Known characters: {char_list}

Extract:
1. DIALOGUE: Speech with speaker. Format: [{{"speaker": "Name", "text": "dialogue"}}]
2. ACTION: Physical actions happening
3. SOUND_EFFECTS: Visible onomatopoeia (BAM, WHOOSH, etc.)
4. EMOTION: Overall emotional tone

Respond ONLY in this JSON format:
{{"dialogue": [{{"speaker": "Character1", "text": "What they say"}}], "action": "description", "sound_effects": ["BAM"], "emotion": "angry"}}

Match speakers to known characters when possible."""}
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
            output_ids = self.vlm.generate(**inputs, max_new_tokens=300, temperature=0.1)

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        return {
            "dialogue": [],
            "action": response[:100],
            "sound_effects": [],
            "emotion": "neutral"
        }

    def generate_dialogue_audio(self, dialogue_entries: list[dict]) -> Optional[np.ndarray]:
        if not dialogue_entries:
            return None

        audio_segments = []

        for entry in dialogue_entries:
            if isinstance(entry, dict):
                speaker = entry.get("speaker", "unknown")
                text = entry.get("text", "")
            else:
                speaker = "unknown"
                text = str(entry)

            if not text or len(text.strip()) < 2:
                continue

            text = text.strip()[:150]
            voice_preset = self._get_voice_for_character(speaker)

            try:
                inputs = self.tts_tokenizer(text, return_tensors="pt", voice_preset=voice_preset)
                input_ids = inputs["input_ids"].to(self.device)

                with torch.no_grad():
                    audio = self.tts.generate(
                        input_ids,
                        do_sample=True,
                        fine_temperature=0.4,
                        coarse_temperature=0.4
                    )

                audio_np = audio.cpu().numpy().squeeze()
                audio_segments.append(audio_np)

                silence = np.zeros(int(self.tts_sample_rate * 0.2))
                audio_segments.append(silence)

            except Exception:
                continue

        if not audio_segments:
            return None

        return np.concatenate(audio_segments)

    def generate_sfx_audio(self, action: str, sound_effects: list[str], emotion: str) -> np.ndarray:
        if sound_effects:
            sfx_text = ", ".join(sound_effects)
            prompt = f"comic book sound effect {sfx_text}, {action}, {emotion} mood"
        elif action:
            prompt = f"comic book sound effect for {action}, {emotion} mood"
        else:
            prompt = f"ambient background sound, {emotion} mood"

        generator = torch.Generator(device=self.device).manual_seed(np.random.randint(0, 2**32))

        with torch.no_grad():
            audio = self.sfx_pipe(
                prompt,
                num_inference_steps=20,
                audio_length_in_s=2.0,
                generator=generator
            ).audios[0]

        return audio

    def mix_audio(self, dialogue_audio: Optional[np.ndarray], sfx_audio: np.ndarray) -> tuple[int, np.ndarray]:
        output_sr = self.tts_sample_rate

        sfx_resampled = torchaudio.functional.resample(
            torch.from_numpy(sfx_audio).unsqueeze(0),
            orig_freq=self.sfx_sample_rate,
            new_freq=output_sr
        ).squeeze().numpy()

        if dialogue_audio is not None and len(dialogue_audio) > 0:
            silence = np.zeros(int(output_sr * 0.3))
            combined = np.concatenate([dialogue_audio, silence, sfx_resampled])
        else:
            combined = sfx_resampled

        combined = combined / (np.abs(combined).max() + 1e-8)
        audio_int16 = (combined * 32767).astype(np.int16)

        return output_sr, audio_int16

    def process_click(self, image: Optional[np.ndarray], evt: gr.SelectData) -> tuple[str, tuple[int, np.ndarray]]:
        if image is None or len(self.pages) == 0:
            return "No image loaded", (24000, np.zeros(24000, dtype=np.int16))

        click_x, click_y = evt.index
        pil_image = self.pages[self.current_page_idx]
        img_w, img_h = pil_image.size

        region_size = min(img_w, img_h) // 3

        x1 = max(0, click_x - region_size // 2)
        y1 = max(0, click_y - region_size // 2)
        x2 = min(img_w, x1 + region_size)
        y2 = min(img_h, y1 + region_size)

        panel_data = self.analyze_panel(pil_image, x1, y1, x2 - x1, y2 - y1)

        dialogue_audio = self.generate_dialogue_audio(panel_data.get("dialogue", []))
        sfx_audio = self.generate_sfx_audio(
            panel_data.get("action", ""),
            panel_data.get("sound_effects", []),
            panel_data.get("emotion", "neutral")
        )

        sample_rate, combined_audio = self.mix_audio(dialogue_audio, sfx_audio)

        dialogue_str = ""
        for d in panel_data.get("dialogue", []):
            if isinstance(d, dict):
                dialogue_str += f"{d.get('speaker', '?')}: \"{d.get('text', '')}\"\n"
            else:
                dialogue_str += f"?: \"{d}\"\n"

        description = (
            f"**Dialogue:**\n{dialogue_str if dialogue_str else 'None'}\n\n"
            f"**Action:** {panel_data.get('action', 'N/A')}\n"
            f"**SFX:** {panel_data.get('sound_effects', [])}\n"
            f"**Emotion:** {panel_data.get('emotion', 'N/A')}"
        )

        return description, (sample_rate, combined_audio)


def build_ui(reader: ComicReader) -> gr.Blocks:
    with gr.Blocks(title="AI Comic Reader") as app:
        gr.Markdown("# AI Comic Reader\nLoad a comic to detect characters and assign voices. Click panels to hear dialogue and sound effects.")

        with gr.Row():
            comic_dropdown = gr.Dropdown(
                choices=reader.list_comics(),
                label="Select Comic",
                interactive=True
            )
            load_btn = gr.Button("Load & Analyze", variant="primary")
            refresh_btn = gr.Button("Refresh")
            reset_btn = gr.Button("Reset")

        with gr.Row():
            character_report = gr.Markdown("*Load a comic to detect characters*")

        with gr.Row():
            page_slider = gr.Slider(0, 0, step=1, label="Page", interactive=True)

        with gr.Row():
            comic_image = gr.Image(label="Comic Page (Click on a panel)")

        with gr.Row():
            description_output = gr.Markdown("*Click a panel to analyze*")

        with gr.Row():
            audio_output = gr.Audio(label="Generated Audio", autoplay=True)

        def refresh_list():
            return gr.update(choices=reader.list_comics())

        def load_comic(filename, progress=gr.Progress()):
            if not filename:
                return None, gr.update(maximum=0, value=0), "*No comic selected*"
            reader.load_comic(filename, progress)
            if len(reader.pages) == 0:
                return None, gr.update(maximum=0, value=0), "*Failed to load comic*"
            max_page = len(reader.pages) - 1

            char_report = "**Detected Characters & Voices:**\n"
            for char, voice in reader.characters.items():
                char_report += f"- {char.title()}: `{voice}`\n"
            if not reader.characters:
                char_report = "*No characters detected*"

            return np.array(reader.pages[0]), gr.update(maximum=max_page, value=0), char_report

        def change_page(page_num):
            if len(reader.pages) == 0:
                return None
            reader.current_page_idx = int(page_num)
            return np.array(reader.pages[reader.current_page_idx])

        def reset_all():
            reader.pages = []
            reader.current_page_idx = 0
            reader.characters = {}
            return (
                gr.update(value=None),
                gr.update(maximum=0, value=0),
                "*Click a panel to analyze*",
                None,
                gr.update(value=None),
                "*Load a comic to detect characters*"
            )

        refresh_btn.click(refresh_list, outputs=[comic_dropdown])
        load_btn.click(
            load_comic,
            inputs=[comic_dropdown],
            outputs=[comic_image, page_slider, character_report]
        )
        page_slider.change(change_page, inputs=[page_slider], outputs=[comic_image])
        comic_image.select(
            reader.process_click,
            inputs=[comic_image],
            outputs=[description_output, audio_output]
        )
        reset_btn.click(
            reset_all,
            outputs=[comic_dropdown, page_slider, description_output, audio_output, comic_image, character_report]
        )

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--tts-model", default="suno/bark-small")
    parser.add_argument("--sfx-model", default="cvssp/audioldm-s-full-v2")
    parser.add_argument("--comics-dir", default="./comics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7777)
    args = parser.parse_args()

    reader = ComicReader(
        vlm_model=args.vlm_model,
        tts_model=args.tts_model,
        sfx_model=args.sfx_model,
        comics_dir=args.comics_dir,
        device=args.device
    )

    app = build_ui(reader)
    app.launch(server_name=args.host, server_port=args.port, share=True)


if __name__ == "__main__":
    main()