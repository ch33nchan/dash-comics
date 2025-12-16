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


VOICE_PROFILES = {
    "male_hero": {"voice": "en-US-GuyNeural", "style": "excited", "rate": "+5%", "pitch": "+0Hz"},
    "male_villain": {"voice": "en-US-ChristopherNeural", "style": "angry", "rate": "-5%", "pitch": "-10Hz"},
    "male_calm": {"voice": "en-GB-RyanNeural", "style": "calm", "rate": "+0%", "pitch": "+0Hz"},
    "male_old": {"voice": "en-US-RogerNeural", "style": "sad", "rate": "-10%", "pitch": "-15Hz"},
    "female_hero": {"voice": "en-US-JennyNeural", "style": "excited", "rate": "+5%", "pitch": "+5Hz"},
    "female_villain": {"voice": "en-US-AriaNeural", "style": "angry", "rate": "-5%", "pitch": "-5Hz"},
    "female_calm": {"voice": "en-GB-SoniaNeural", "style": "calm", "rate": "+0%", "pitch": "+0Hz"},
    "female_old": {"voice": "en-US-JaneNeural", "style": "sad", "rate": "-10%", "pitch": "+0Hz"},
    "child": {"voice": "en-US-AnaNeural", "style": "cheerful", "rate": "+10%", "pitch": "+20Hz"},
    "narrator": {"voice": "en-US-DavisNeural", "style": "documentary-narration", "rate": "-5%", "pitch": "-5Hz"},
}

EMOTION_STYLES = {
    "angry": {"style": "angry", "rate": "+10%", "pitch": "+10Hz"},
    "sad": {"style": "sad", "rate": "-15%", "pitch": "-10Hz"},
    "happy": {"style": "cheerful", "rate": "+10%", "pitch": "+10Hz"},
    "excited": {"style": "excited", "rate": "+15%", "pitch": "+15Hz"},
    "scared": {"style": "terrified", "rate": "+20%", "pitch": "+20Hz"},
    "tense": {"style": "serious", "rate": "-5%", "pitch": "-5Hz"},
    "calm": {"style": "calm", "rate": "-5%", "pitch": "+0Hz"},
    "shouting": {"style": "shouting", "rate": "+10%", "pitch": "+20Hz"},
    "whispering": {"style": "whispering", "rate": "-20%", "pitch": "-10Hz"},
    "neutral": {"style": "neutral", "rate": "+0%", "pitch": "+0Hz"},
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
        self.output_sample_rate = 24000

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

        progress(0.1, desc="Analyzing all characters across comic...")
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

    def _analyze_all_characters(self, progress) -> str:
        if not self.pages:
            return "No pages loaded"

        all_characters = {}
        total_pages = len(self.pages)

        self.characters["NARRATOR"] = {
            "profile": "narrator",
            "description": "Story narrator for caption boxes",
            "gender": "male",
            "role": "narrator"
        }

        for i, page in enumerate(self.pages):
            progress_val = 0.1 + (0.8 * (i + 1) / total_pages)
            progress(progress_val, desc=f"Scanning page {i + 1}/{total_pages} for characters...")

            prompt = """Look at this comic page very carefully. I need you to identify EVERY character visible.

For EACH character you can see, tell me:
1. Their NAME if visible anywhere (speech bubbles, labels, mentioned by others)
2. If no name visible, give a DESCRIPTIVE NAME based on appearance (e.g., "Red-suited hero", "Masked villain")
3. GENDER: male/female
4. AGE: child/young/adult/old
5. ROLE: hero/villain/sidekick/civilian/narrator
6. Any distinctive VISUAL features

Also identify:
- NARRATOR boxes (rectangular caption boxes with story text)
- Any text labels or character introductions

Return ONLY valid JSON:
{
  "characters": [
    {
      "name": "Spider-Man",
      "gender": "male", 
      "age": "young",
      "role": "hero",
      "features": "red and blue suit with web pattern, mask"
    },
    {
      "name": "Blonde woman in lab coat",
      "gender": "female",
      "age": "adult", 
      "role": "civilian",
      "features": "blonde hair, white lab coat, glasses"
    }
  ],
  "has_narrator_boxes": true
}

Be EXHAUSTIVE - list every single character visible, even in background."""

            response = self._vlm_query(page, prompt, max_tokens=600)

            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    for char in data.get("characters", []):
                        name = char.get("name", "").strip()
                        if not name:
                            continue
                        name_key = name.lower()
                        if name_key not in all_characters:
                            all_characters[name_key] = {
                                "display_name": name,
                                "gender": char.get("gender", "unknown"),
                                "age": char.get("age", "adult"),
                                "role": char.get("role", "civilian"),
                                "features": char.get("features", "")
                            }
            except json.JSONDecodeError:
                continue

        for name_key, info in all_characters.items():
            gender = info.get("gender", "unknown").lower()
            age = info.get("age", "adult").lower()
            role = info.get("role", "civilian").lower()

            if role == "hero":
                profile = f"{gender}_hero" if gender in ["male", "female"] else "male_hero"
            elif role == "villain":
                profile = f"{gender}_villain" if gender in ["male", "female"] else "male_villain"
            elif age == "child":
                profile = "child"
            elif age == "old":
                profile = f"{gender}_old" if gender in ["male", "female"] else "male_old"
            else:
                profile = f"{gender}_calm" if gender in ["male", "female"] else "male_calm"

            self.characters[info["display_name"]] = {
                "profile": profile,
                "description": info.get("features", ""),
                "gender": gender,
                "role": role
            }

        report = f"**Found {len(self.characters)} Characters:**\n\n"
        for name, info in self.characters.items():
            profile = VOICE_PROFILES.get(info["profile"], VOICE_PROFILES["male_calm"])
            report += f"- **{name}** ({info.get('role', 'unknown')}): `{profile['voice']}` - {info.get('description', '')[:40]}\n"

        return report

    def _get_voice_config(self, speaker: str, emotion: str) -> dict:
        speaker_upper = speaker.strip().upper()

        if speaker_upper in ["NARRATOR", "NARRATION", "CAPTION"]:
            base_profile = VOICE_PROFILES["narrator"].copy()
        else:
            char_info = None
            for char_name, info in self.characters.items():
                if char_name.upper() == speaker_upper or speaker_upper in char_name.upper():
                    char_info = info
                    break

            if char_info:
                base_profile = VOICE_PROFILES.get(char_info["profile"], VOICE_PROFILES["male_calm"]).copy()
            else:
                base_profile = VOICE_PROFILES["male_calm"].copy()

        emotion_lower = emotion.lower() if emotion else "neutral"
        if emotion_lower in EMOTION_STYLES:
            emotion_config = EMOTION_STYLES[emotion_lower]
            base_profile["style"] = emotion_config.get("style", base_profile.get("style", "neutral"))
            base_profile["rate"] = emotion_config.get("rate", base_profile.get("rate", "+0%"))
            base_profile["pitch"] = emotion_config.get("pitch", base_profile.get("pitch", "+0Hz"))

        return base_profile

    def analyze_panel(self, image: Image.Image, x: int, y: int, w: int, h: int) -> dict:
        region = image.crop((x, y, x + w, y + h))
        char_names = [name for name in self.characters.keys() if name != "NARRATOR"]

        prompt = f"""Analyze this comic panel like a movie director. Extract EVERYTHING for voice acting.

Known characters: {char_names}

I need you to identify IN ORDER OF OCCURRENCE (like reading the panel):

1. NARRATION: Any rectangular caption boxes (story narration, setting descriptions, time stamps like "Meanwhile...", "Later that day...")

2. DIALOGUE: Every speech bubble. For each one:
   - WHO is speaking (match to known characters or describe them)
   - WHAT they say (exact text)
   - HOW they say it (emotion: angry, scared, happy, sad, excited, calm, shouting, whispering)

3. THOUGHTS: Cloud-shaped thought bubbles
   - WHO is thinking
   - WHAT they're thinking
   - The emotion

4. SOUND EFFECTS: All visible onomatopoeia (POW, BAM, CRASH, WHOOSH, etc.)
   - What sound it represents
   - When it occurs (during which dialogue, or standalone)

5. ACTION: Physical actions happening that need sound (punching, running, door slamming)

Return this EXACT JSON structure:
{{
  "sequence": [
    {{"type": "narration", "text": "Meanwhile, at the Daily Bugle...", "emotion": "calm"}},
    {{"type": "dialogue", "speaker": "Spider-Man", "text": "I've got to stop him!", "emotion": "excited"}},
    {{"type": "sfx", "text": "CRASH", "action": "window breaking"}},
    {{"type": "dialogue", "speaker": "Villain", "text": "You're too late!", "emotion": "angry"}},
    {{"type": "thought", "speaker": "Spider-Man", "text": "This doesn't look good...", "emotion": "scared"}}
  ],
  "ambient_action": "fight scene with breaking glass",
  "overall_emotion": "tense"
}}

CRITICAL: 
- Put items in the ORDER they should be heard (top-to-bottom, left-to-right)
- Don't miss ANY text visible in the panel
- Sound effects should be placed WHERE they occur in the sequence
- Match speakers to the character list when possible"""

        response = self._vlm_query(region, prompt, max_tokens=700)
        print(f"VLM Response: {response}")

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")

        return {
            "sequence": [],
            "ambient_action": "",
            "overall_emotion": "neutral"
        }

    async def _generate_speech_with_emotion(self, text: str, voice_config: dict) -> Optional[np.ndarray]:
        if not text or len(text.strip()) < 2:
            return None

        voice = voice_config.get("voice", "en-US-GuyNeural")
        rate = voice_config.get("rate", "+0%")
        pitch = voice_config.get("pitch", "+0Hz")

        ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
    <voice name="{voice}">
        <prosody rate="{rate}" pitch="{pitch}">
            {text}
        </prosody>
    </voice>
</speak>"""

        try:
            communicate = edge_tts.Communicate(ssml, voice)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
                await communicate.save(tmp.name)
                audio, sr = sf.read(tmp.name)

                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)

                if sr != self.output_sample_rate:
                    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                    audio = torchaudio.functional.resample(
                        audio_tensor, orig_freq=sr, new_freq=self.output_sample_rate
                    ).squeeze().numpy()

                return audio
        except Exception as e:
            print(f"TTS Error for '{text[:30]}...': {e}")
            try:
                communicate = edge_tts.Communicate(text, voice)
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
                    await communicate.save(tmp.name)
                    audio, sr = sf.read(tmp.name)
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                    if sr != self.output_sample_rate:
                        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                        audio = torchaudio.functional.resample(
                            audio_tensor, orig_freq=sr, new_freq=self.output_sample_rate
                        ).squeeze().numpy()
                    return audio
            except Exception as e2:
                print(f"Fallback TTS also failed: {e2}")
                return None

    def _generate_sfx(self, action: str, sfx_texts: list[str]) -> Optional[np.ndarray]:
        if not action and not sfx_texts:
            return None

        if sfx_texts:
            sfx_str = " ".join(sfx_texts)
            prompt = f"cinematic sound effect: {sfx_str}, {action}, dramatic impact, movie quality"
        else:
            prompt = f"cinematic sound effect: {action}, dramatic, movie quality"

        generator = torch.Generator(device=self.device).manual_seed(np.random.randint(0, 2**32))

        with torch.no_grad():
            audio = self.sfx_pipe(
                prompt,
                num_inference_steps=25,
                audio_length_in_s=2.0,
                generator=generator
            ).audios[0]

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        audio_resampled = torchaudio.functional.resample(
            audio_tensor, orig_freq=self.sfx_sample_rate, new_freq=self.output_sample_rate
        ).squeeze().numpy()

        return audio_resampled

    def _mix_like_movie(
        self,
        speech_segments: list[tuple[str, np.ndarray]],
        sfx_audio: Optional[np.ndarray],
        sfx_positions: list[int]
    ) -> np.ndarray:
        if not speech_segments and sfx_audio is None:
            return np.zeros(self.output_sample_rate, dtype=np.float32)

        total_speech = []
        segment_starts = []
        current_pos = 0

        for seg_type, audio in speech_segments:
            segment_starts.append(current_pos)
            total_speech.append(audio)
            pause = int(self.output_sample_rate * 0.15)
            total_speech.append(np.zeros(pause))
            current_pos += len(audio) + pause

        if total_speech:
            speech_track = np.concatenate(total_speech)
        else:
            speech_track = np.zeros(self.output_sample_rate)

        if sfx_audio is not None and len(sfx_audio) > 0:
            total_length = max(len(speech_track), len(sfx_audio) + int(self.output_sample_rate * 0.5))
            mixed = np.zeros(total_length, dtype=np.float32)

            mixed[:len(speech_track)] = speech_track

            if sfx_positions and segment_starts:
                for sfx_pos in sfx_positions:
                    if sfx_pos < len(segment_starts):
                        start_sample = segment_starts[sfx_pos]
                    else:
                        start_sample = len(speech_track) - len(sfx_audio) // 2
            else:
                start_sample = max(0, len(speech_track) // 2 - len(sfx_audio) // 2)

            start_sample = max(0, min(start_sample, total_length - len(sfx_audio)))
            end_sample = start_sample + len(sfx_audio)

            sfx_volume = 0.4
            mixed[start_sample:end_sample] += sfx_audio * sfx_volume

            mixed = mixed / (np.abs(mixed).max() + 1e-8)
        else:
            mixed = speech_track
            if np.abs(mixed).max() > 0:
                mixed = mixed / (np.abs(mixed).max() + 1e-8)

        return mixed

    def process_click(self, image: Optional[np.ndarray], evt: gr.SelectData) -> tuple[str, tuple[int, np.ndarray]]:
        if image is None or len(self.pages) == 0:
            return "No image loaded", (self.output_sample_rate, np.zeros(self.output_sample_rate, dtype=np.int16))

        click_x, click_y = evt.index
        pil_image = self.pages[self.current_page_idx]
        img_w, img_h = pil_image.size

        region_size = min(img_w, img_h) // 3
        x1 = max(0, click_x - region_size // 2)
        y1 = max(0, click_y - region_size // 2)
        x2 = min(img_w, x1 + region_size)
        y2 = min(img_h, y1 + region_size)

        print(f"\n{'='*50}")
        print(f"Analyzing panel at [{x1},{y1}] to [{x2},{y2}]")
        print(f"{'='*50}")

        panel_data = self.analyze_panel(pil_image, x1, y1, x2 - x1, y2 - y1)
        sequence = panel_data.get("sequence", [])
        ambient_action = panel_data.get("ambient_action", "")
        overall_emotion = panel_data.get("overall_emotion", "neutral")

        print(f"Found {len(sequence)} items in sequence")
        print(f"Ambient action: {ambient_action}")
        print(f"Overall emotion: {overall_emotion}")

        async def generate_audio():
            speech_segments = []
            sfx_texts = []
            sfx_positions = []

            for i, item in enumerate(sequence):
                item_type = item.get("type", "")
                text = item.get("text", "")
                emotion = item.get("emotion", overall_emotion)

                print(f"  [{i}] {item_type}: {text[:50]}... (emotion: {emotion})")

                if item_type in ["narration", "dialogue", "thought"]:
                    speaker = item.get("speaker", "NARRATOR") if item_type != "narration" else "NARRATOR"

                    if item_type == "thought":
                        text = f"thinking to themselves... {text}"

                    voice_config = self._get_voice_config(speaker, emotion)
                    print(f"      Voice: {voice_config['voice']}, Style: {voice_config.get('style')}")

                    audio = await self._generate_speech_with_emotion(text, voice_config)
                    if audio is not None and len(audio) > 0:
                        speech_segments.append((item_type, audio))
                        print(f"      Generated {len(audio)/self.output_sample_rate:.2f}s of audio")

                elif item_type == "sfx":
                    sfx_texts.append(text)
                    sfx_positions.append(len(speech_segments))
                    print(f"      SFX queued: {text}")

            sfx_audio = None
            if sfx_texts or ambient_action:
                print(f"Generating SFX for: {sfx_texts} / {ambient_action}")
                sfx_audio = self._generate_sfx(ambient_action, sfx_texts)

            return speech_segments, sfx_audio, sfx_positions

        speech_segments, sfx_audio, sfx_positions = asyncio.run(generate_audio())

        print(f"Mixing {len(speech_segments)} speech segments with SFX...")
        mixed_audio = self._mix_like_movie(speech_segments, sfx_audio, sfx_positions)

        audio_int16 = (mixed_audio * 32767).astype(np.int16)

        description_parts = ["### Panel Sequence\n"]
        for i, item in enumerate(sequence):
            item_type = item.get("type", "unknown")
            if item_type == "narration":
                description_parts.append(f"**[NARRATOR]**: _{item.get('text', '')}_\n")
            elif item_type == "dialogue":
                speaker = item.get("speaker", "Unknown")
                emotion = item.get("emotion", "")
                description_parts.append(f"**{speaker}** ({emotion}): \"{item.get('text', '')}\"\n")
            elif item_type == "thought":
                speaker = item.get("speaker", "Unknown")
                description_parts.append(f"**{speaker}** 💭: _{item.get('text', '')}_\n")
            elif item_type == "sfx":
                description_parts.append(f"**[SFX]**: 💥 {item.get('text', '')} ({item.get('action', '')})\n")

        description_parts.append(f"\n**Scene**: {ambient_action}")
        description_parts.append(f"\n**Mood**: {overall_emotion}")

        return "\n".join(description_parts), (self.output_sample_rate, audio_int16)


def build_ui(reader: ComicReader) -> gr.Blocks:
    with gr.Blocks(title="AI Comic Reader", theme=gr.themes.Soft()) as app:
        gr.Markdown("""# 🎬 AI Comic Reader - Movie Mode
        
Load a comic to detect ALL characters with unique voices. Click panels to experience them like a movie - with emotional dialogue, narration, and sound effects mixed together.""")

        with gr.Row():
            comic_dropdown = gr.Dropdown(
                choices=reader.list_comics(),
                label="Select Comic",
                interactive=True
            )
            load_btn = gr.Button("🎬 Load & Analyze Characters", variant="primary")
            refresh_btn = gr.Button("🔄 Refresh")
            reset_btn = gr.Button("❌ Reset", variant="stop")

        with gr.Row():
            character_report = gr.Markdown("*Load a comic to detect characters and assign emotional voices*")

        with gr.Row():
            page_slider = gr.Slider(0, 0, step=1, label="Page", interactive=True)

        with gr.Row():
            comic_image = gr.Image(label="📖 Comic Page - Click any panel to bring it to life!")

        with gr.Row():
            with gr.Column(scale=1):
                description_output = gr.Markdown("*Click a panel to see the script and hear it performed*")
            with gr.Column(scale=1):
                audio_output = gr.Audio(label="🎧 Movie Audio", autoplay=True)

        def refresh_list():
            return gr.update(choices=reader.list_comics())

        def load_comic(filename, progress=gr.Progress()):
            if not filename:
                return None, gr.update(maximum=0, value=0), "*No comic selected*"

            reader.load_comic(filename, progress)

            if len(reader.pages) == 0:
                return None, gr.update(maximum=0, value=0), "*Failed to load comic*"

            max_page = len(reader.pages) - 1

            char_report = f"### 🎭 Cast of Characters ({len(reader.characters)} found)\n\n"
            for name, info in reader.characters.items():
                profile = VOICE_PROFILES.get(info["profile"], VOICE_PROFILES["male_calm"])
                role_emoji = {"hero": "🦸", "villain": "🦹", "narrator": "📖", "civilian": "👤"}.get(info.get("role", ""), "👤")
                char_report += f"- {role_emoji} **{name}**: `{profile['voice']}` ({info.get('role', 'unknown')})\n"

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
                "*Click a panel to see the script*",
                None,
                gr.update(value=None),
                "*Load a comic to detect characters*"
            )

        refresh_btn.click(refresh_list, outputs=[comic_dropdown])
        load_btn.click(load_comic, inputs=[comic_dropdown], outputs=[comic_image, page_slider, character_report])
        page_slider.change(change_page, inputs=[page_slider], outputs=[comic_image])
        comic_image.select(reader.process_click, inputs=[comic_image], outputs=[description_output, audio_output])
        reset_btn.click(reset_all, outputs=[comic_dropdown, page_slider, description_output, audio_output, comic_image, character_report])

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