import argparse
import asyncio
import io
import json
import re
import tempfile
from pathlib import Path
from typing import Optional

import edge_tts
import fitz
import gradio as gr
import numpy as np
import rarfile
import soundfile as sf
import torch
import torchaudio
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from diffusers import AudioLDMPipeline


EDGE_VOICES = {
    "male_1": "en-US-GuyNeural",
    "male_2": "en-US-ChristopherNeural",
    "male_3": "en-GB-RyanNeural",
    "male_4": "en-AU-WilliamNeural",
    "female_1": "en-US-JennyNeural",
    "female_2": "en-US-AriaNeural",
    "female_3": "en-GB-SoniaNeural",
    "female_4": "en-AU-NatashaNeural",
    "narrator": "en-US-DavisNeural",
    "old_male": "en-US-RogerNeural",
    "old_female": "en-US-JaneNeural",
    "child": "en-US-AnaNeural",
}


class ComicReader:
    def __init__(
        self,
        vlm_model: str,
        sfx_model: str,
        comics_dir: str,
        device: str = "cuda"
    ):
        self.device = device
        self.comics_dir = Path(comics_dir)
        self.pages: list[Image.Image] = []
        self.current_page_idx: int = 0
        self.characters: dict[str, dict] = {}

        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            vlm_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_model)

        self.sfx_pipe = AudioLDMPipeline.from_pretrained(
            sfx_model,
            torch_dtype=torch.float16
        ).to(device)
        self.sfx_sample_rate = 16000
        self.tts_sample_rate = 24000

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

        progress(0, desc="Loading pages...")
        if filename.lower().endswith(".cbr"):
            self._load_cbr(filepath)
        elif filename.lower().endswith(".pdf"):
            self._load_pdf(filepath)

        self.current_page_idx = 0

        progress(0.2, desc="Analyzing all characters...")
        char_report = self._analyze_all_characters(progress)

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

    def _vlm_query(self, image: Image.Image, prompt: str, max_tokens: int = 300) -> str:
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
            output_ids = self.vlm.generate(**inputs, max_new_tokens=max_tokens, temperature=0.1)

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def _analyze_all_characters(self, progress) -> str:
        if not self.pages:
            return "No pages loaded"

        all_characters = {}
        total_pages = len(self.pages)
        
        self.characters["NARRATOR"] = {
            "voice": EDGE_VOICES["narrator"],
            "description": "Story narrator, caption boxes",
            "gender": "male"
        }

        for i, page in enumerate(self.pages):
            progress_val = 0.2 + (0.7 * (i + 1) / total_pages)
            progress(progress_val, desc=f"Scanning page {i + 1}/{total_pages}...")

            prompt = """Analyze this comic book page carefully. Identify ALL characters present.

For EACH character, provide:
1. Name (if shown in text/dialogue) OR a descriptive identifier (e.g., "Man in red cape", "Blonde woman")
2. Gender (male/female/unknown)
3. Estimated age (child/young/adult/old)
4. Brief visual description

Also note if there are NARRATION BOXES (rectangular boxes with story text, not speech bubbles).

Respond in this exact JSON format:
{
  "characters": [
    {"name": "Spider-Man", "gender": "male", "age": "young", "description": "Red and blue suit, web pattern"},
    {"name": "Woman in black dress", "gender": "female", "age": "adult", "description": "Long dark hair, elegant dress"}
  ],
  "has_narration": true
}

Be thorough - identify EVERY visible character, even background ones."""

            response = self._vlm_query(page, prompt, max_tokens=500)

            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    for char in data.get("characters", []):
                        name = char.get("name", "Unknown").strip()
                        if name and name.lower() not in [c.lower() for c in all_characters.keys()]:
                            all_characters[name] = {
                                "gender": char.get("gender", "unknown"),
                                "age": char.get("age", "adult"),
                                "description": char.get("description", "")
                            }
            except json.JSONDecodeError:
                continue

        voice_pool = {
            ("male", "child"): EDGE_VOICES["child"],
            ("male", "young"): EDGE_VOICES["male_1"],
            ("male", "adult"): EDGE_VOICES["male_2"],
            ("male", "old"): EDGE_VOICES["old_male"],
            ("female", "child"): EDGE_VOICES["child"],
            ("female", "young"): EDGE_VOICES["female_1"],
            ("female", "adult"): EDGE_VOICES["female_2"],
            ("female", "old"): EDGE_VOICES["old_female"],
        }

        male_voices = [EDGE_VOICES["male_1"], EDGE_VOICES["male_2"], EDGE_VOICES["male_3"], EDGE_VOICES["male_4"]]
        female_voices = [EDGE_VOICES["female_1"], EDGE_VOICES["female_2"], EDGE_VOICES["female_3"], EDGE_VOICES["female_4"]]
        male_idx, female_idx = 0, 0

        for name, info in all_characters.items():
            gender = info.get("gender", "unknown").lower()
            age = info.get("age", "adult").lower()

            key = (gender, age)
            if key in voice_pool:
                voice = voice_pool[key]
            elif gender == "male":
                voice = male_voices[male_idx % len(male_voices)]
                male_idx += 1
            elif gender == "female":
                voice = female_voices[female_idx % len(female_voices)]
                female_idx += 1
            else:
                voice = male_voices[male_idx % len(male_voices)]
                male_idx += 1

            self.characters[name] = {
                "voice": voice,
                "description": info.get("description", ""),
                "gender": gender
            }

        report = f"**Detected {len(self.characters)} Characters:**\n\n"
        for name, info in self.characters.items():
            report += f"- **{name}**: {info['voice']} ({info.get('description', 'N/A')[:50]})\n"

        return report

    def _get_voice_for_speaker(self, speaker: str) -> str:
        speaker_clean = speaker.strip().upper()
        
        if speaker_clean in ["NARRATOR", "NARRATION", "CAPTION", "BOX"]:
            return EDGE_VOICES["narrator"]

        for char_name, info in self.characters.items():
            if char_name.upper() == speaker_clean:
                return info["voice"]

        for char_name, info in self.characters.items():
            if char_name.upper() in speaker_clean or speaker_clean in char_name.upper():
                return info["voice"]

        return EDGE_VOICES["male_1"]

    def analyze_panel(self, image: Image.Image, x: int, y: int, w: int, h: int) -> dict:
        region = image.crop((x, y, x + w, y + h))

        char_names = list(self.characters.keys())

        prompt = f"""Analyze this comic panel in detail.

Known characters in this comic: {char_names}

Extract ALL of the following:

1. NARRATION: Text in rectangular caption boxes (story narration, not dialogue)
2. DIALOGUE: Speech bubbles - identify WHO is speaking and WHAT they say
3. THOUGHT: Thought bubbles (usually cloud-shaped)
4. ACTION: What physical actions are happening
5. SOUND_EFFECTS: Visible onomatopoeia text (BAM, CRASH, WHOOSH, etc.)
6. EMOTION: Overall emotional tone of the scene

IMPORTANT: 
- Match speakers to the known characters list when possible
- If a caption/narration box exists, mark speaker as "NARRATOR"
- Include ALL text visible in the panel

Respond ONLY in this exact JSON format:
{{
  "narration": "Any narrator caption box text here",
  "dialogue": [
    {{"speaker": "Character Name", "text": "What they say"}},
    {{"speaker": "NARRATOR", "text": "Caption box text"}}
  ],
  "thought": [{{"speaker": "Character Name", "text": "What they think"}}],
  "action": "Description of physical actions",
  "sound_effects": ["BAM", "CRASH"],
  "emotion": "tense"
}}"""

        response = self._vlm_query(region, prompt, max_tokens=500)

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                all_dialogue = []
                
                if result.get("narration"):
                    all_dialogue.append({"speaker": "NARRATOR", "text": result["narration"]})
                
                for d in result.get("dialogue", []):
                    if isinstance(d, dict) and d.get("text"):
                        all_dialogue.append(d)
                
                for t in result.get("thought", []):
                    if isinstance(t, dict) and t.get("text"):
                        t["text"] = f"thinking... {t['text']}"
                        all_dialogue.append(t)
                
                result["all_speech"] = all_dialogue
                return result
        except json.JSONDecodeError:
            pass

        return {
            "narration": "",
            "dialogue": [],
            "thought": [],
            "action": response[:100] if response else "Unable to analyze",
            "sound_effects": [],
            "emotion": "neutral",
            "all_speech": []
        }

    async def _generate_speech_async(self, text: str, voice: str) -> Optional[np.ndarray]:
        if not text or len(text.strip()) < 2:
            return None

        try:
            communicate = edge_tts.Communicate(text, voice)
            
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
                await communicate.save(tmp.name)
                audio, sr = sf.read(tmp.name)
                
                if sr != self.tts_sample_rate:
                    audio_tensor = torch.from_numpy(audio).float()
                    if audio_tensor.dim() == 1:
                        audio_tensor = audio_tensor.unsqueeze(0)
                    else:
                        audio_tensor = audio_tensor.T
                    audio_resampled = torchaudio.functional.resample(
                        audio_tensor, orig_freq=sr, new_freq=self.tts_sample_rate
                    )
                    audio = audio_resampled.squeeze().numpy()
                
                return audio
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

    def generate_dialogue_audio(self, speech_entries: list[dict]) -> Optional[np.ndarray]:
        if not speech_entries:
            return None

        async def generate_all():
            audio_segments = []
            
            for entry in speech_entries:
                speaker = entry.get("speaker", "Unknown")
                text = entry.get("text", "")
                
                if not text or len(text.strip()) < 2:
                    continue

                voice = self._get_voice_for_speaker(speaker)
                print(f"Generating speech: [{speaker}] -> {voice}: {text[:50]}...")
                
                audio = await self._generate_speech_async(text, voice)
                
                if audio is not None and len(audio) > 0:
                    audio_segments.append(audio)
                    silence = np.zeros(int(self.tts_sample_rate * 0.3))
                    audio_segments.append(silence)

            if audio_segments:
                return np.concatenate(audio_segments)
            return None

        return asyncio.run(generate_all())

    def generate_sfx_audio(self, action: str, sound_effects: list[str], emotion: str) -> Optional[np.ndarray]:
        if not sound_effects and not action:
            return None

        if sound_effects:
            sfx_text = ", ".join(sound_effects)
            prompt = f"comic book sound effect {sfx_text}, {action}, {emotion} mood, dramatic"
        else:
            prompt = f"comic book action sound for {action}, {emotion} mood"

        generator = torch.Generator(device=self.device).manual_seed(np.random.randint(0, 2**32))

        with torch.no_grad():
            audio = self.sfx_pipe(
                prompt,
                num_inference_steps=20,
                audio_length_in_s=1.5,
                generator=generator
            ).audios[0]

        return audio

    def mix_audio(self, dialogue_audio: Optional[np.ndarray], sfx_audio: Optional[np.ndarray]) -> tuple[int, np.ndarray]:
        output_sr = self.tts_sample_rate
        segments = []

        if dialogue_audio is not None and len(dialogue_audio) > 0:
            segments.append(dialogue_audio)

        if sfx_audio is not None and len(sfx_audio) > 0:
            sfx_resampled = torchaudio.functional.resample(
                torch.from_numpy(sfx_audio).unsqueeze(0),
                orig_freq=self.sfx_sample_rate,
                new_freq=output_sr
            ).squeeze().numpy()
            
            if segments:
                silence = np.zeros(int(output_sr * 0.2))
                segments.append(silence)
            segments.append(sfx_resampled)

        if not segments:
            return output_sr, np.zeros(output_sr, dtype=np.int16)

        combined = np.concatenate(segments)
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

        print(f"\n=== Analyzing region [{x1},{y1}] to [{x2},{y2}] ===")
        panel_data = self.analyze_panel(pil_image, x1, y1, x2 - x1, y2 - y1)
        print(f"Panel data: {json.dumps(panel_data, indent=2)}")

        all_speech = panel_data.get("all_speech", [])
        print(f"Speech entries to generate: {len(all_speech)}")
        
        dialogue_audio = self.generate_dialogue_audio(all_speech)
        
        sfx_audio = self.generate_sfx_audio(
            panel_data.get("action", ""),
            panel_data.get("sound_effects", []),
            panel_data.get("emotion", "neutral")
        )

        sample_rate, combined_audio = self.mix_audio(dialogue_audio, sfx_audio)

        narration = panel_data.get("narration", "")
        dialogue_str = ""
        for d in panel_data.get("dialogue", []):
            if isinstance(d, dict):
                dialogue_str += f"**{d.get('speaker', '?')}**: \"{d.get('text', '')}\"\n"
        
        thought_str = ""
        for t in panel_data.get("thought", []):
            if isinstance(t, dict):
                thought_str += f"**{t.get('speaker', '?')}** (thinking): \"{t.get('text', '')}\"\n"

        description = f"""### Panel Analysis

**Narration:** {narration if narration else 'None'}

**Dialogue:**
{dialogue_str if dialogue_str else 'None'}

**Thoughts:**
{thought_str if thought_str else 'None'}

**Action:** {panel_data.get('action', 'N/A')}

**Sound Effects:** {', '.join(panel_data.get('sound_effects', [])) or 'None'}

**Emotion:** {panel_data.get('emotion', 'N/A')}"""

        return description, (sample_rate, combined_audio)


def build_ui(reader: ComicReader) -> gr.Blocks:
    with gr.Blocks(title="AI Comic Reader", theme=gr.themes.Soft()) as app:
        gr.Markdown("""# AI Comic Reader
        
Load a comic to detect ALL characters and assign unique voices. Click on any panel to hear dialogue, narration, and sound effects.""")

        with gr.Row():
            comic_dropdown = gr.Dropdown(
                choices=reader.list_comics(),
                label="Select Comic",
                interactive=True
            )
            load_btn = gr.Button("Load & Analyze All Characters", variant="primary")
            refresh_btn = gr.Button("Refresh List")
            reset_btn = gr.Button("Reset", variant="stop")

        with gr.Row():
            character_report = gr.Markdown("*Load a comic to detect characters and assign voices*")

        with gr.Row():
            page_slider = gr.Slider(0, 0, step=1, label="Page", interactive=True)

        with gr.Row():
            comic_image = gr.Image(label="Comic Page - Click anywhere to analyze that region")

        with gr.Row():
            with gr.Column():
                description_output = gr.Markdown("*Click a panel to analyze and hear*")
            with gr.Column():
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

            char_report = f"**Detected {len(reader.characters)} Characters:**\n\n"
            for name, info in reader.characters.items():
                char_report += f"- **{name}**: `{info['voice']}`\n"

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
                "*Load a comic to detect characters and assign voices*"
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
    parser.add_argument("--sfx-model", default="cvssp/audioldm-s-full-v2")
    parser.add_argument("--comics-dir", default="./comics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7777)
    args = parser.parse_args()

    reader = ComicReader(
        vlm_model=args.vlm_model,
        sfx_model=args.sfx_model,
        comics_dir=args.comics_dir,
        device=args.device
    )

    app = build_ui(reader)
    app.launch(server_name=args.host, server_port=args.port, share=True)


if __name__ == "__main__":
    main()