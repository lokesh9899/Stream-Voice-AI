import torch

_real_torch_load = torch.load
def _torch_load_cpu(*args, **kwargs):
    if 'map_location' not in kwargs or kwargs['map_location'] is None:
        kwargs['map_location'] = torch.device('cpu')
    return _real_torch_load(*args, **kwargs)

torch.load = _torch_load_cpu

import os
import time
import sys
import re
import numpy as np
import sounddevice as sd
import httpx
import json
from dotenv import load_dotenv
import whisperx
from chatterbox.tts import ChatterboxTTS

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY in .env")

SAMPLE_RATE       = 16000
CHANNELS          = 1
VOLUME_THRESHOLD  = 0.02
MAX_DURATION      = 10     
SILENCE_TIMEOUT   = 0.5    
MIN_SPEECH_DURATION = 0.2 

class AI_Assistant:
    def __init__(self):
        self.full_transcript = [{
            "role": "system",
            "content": """
You are a friendly, natural-sounding voice assistant.

Use these speech-control tags in your replies:
- <break> for pauses
- <rate level='slow'>...</rate> to speak slowly
- <pitch level='high'>...</pitch> to sound cheerful
- <emphasis level='strong'>...</emphasis> to highlight key parts

Always speak casually and naturally. Use contractions like "you're" and "let's", and add <break> after greetings or long sentences.

If the user asks to switch to Japanese, do so; if they ask for English, switch back.
"""
        }]
        self.language = "english"
        self.openrouter_api_key = OPENROUTER_API_KEY

        self.device ="cpu"
        print(f"Loading WhisperX model on {self.device}...")
        self.whisper_model = whisperx.load_model(
            "medium",
            device=self.device,
            compute_type=("float16" if self.device=="cuda" else "float32")
        )
        print("WhisperX model loaded")

        print("Loading ChatterboxTTS...")
        self.tts_model = ChatterboxTTS.from_pretrained(device=self.device)
        print("ChatterboxTTS loaded\n")

    def record_with_vad(self):
        print("Listening (VAD enabled)... speak now.")
        frames, total, silence = [], 0, 0
        silence_frames = int(SAMPLE_RATE * SILENCE_TIMEOUT)
        min_frames     = int(SAMPLE_RATE * MIN_SPEECH_DURATION)

        def callback(indata, frame_count, time_info, status):
            nonlocal frames, total, silence
            amp = np.linalg.norm(indata) * 10
            if amp > VOLUME_THRESHOLD:
                frames.append(indata.copy())
                total += frame_count
                silence = 0
            elif total >= min_frames:
                silence += frame_count
                if silence >= silence_frames:
                    raise sd.CallbackStop()

        try:
            with sd.InputStream(callback=callback, channels=CHANNELS, samplerate=SAMPLE_RATE):
                sd.sleep(int(MAX_DURATION * 1000))
        except sd.CallbackStop:
            pass

        if total < min_frames:
            print("No sufficient speech detected.")
            return None

        audio_np = np.concatenate(frames, axis=0) 
        return audio_np

    def transcribe(self, audio_np):
        if audio_np.ndim==2 and audio_np.shape[1]==1:
            audio_np = audio_np.squeeze(1)
        audio_np = audio_np.astype(np.float32)
        t0 = time.time()
        result = self.whisper_model.transcribe(audio_np, batch_size=8)
        duration = time.time() - t0
        text = result.get("text","").strip()
        print(f"Transcribed in {duration:.1f}s: {text or '[no speech]'}")
        return text

    def generate_ai_response(self, user_input):
        print(f"\n👤 User: {user_input}\n")
        inp = user_input.lower()
        if inp in ["exit","stop","quit"]:
            self.generate_audio("Thank you for chatting. Goodbye!")
            sys.exit(0)
        if "japanese" in inp or "日本語" in inp:
            self.language = "japanese"
            self.full_transcript.append({"role":"user","content":user_input})
            self.generate_audio("わかりました。これからは日本語でお話しします。")
            return
        if "english" in inp or "英語" in inp:
            self.language = "english"
            self.full_transcript.append({"role":"user","content":user_input})
            self.generate_audio("Alright, I'll continue in English.")
            return

        self.full_transcript.append({"role":"user","content":user_input})
        reminder = ("From now on, reply only in Japanese using friendly tone and speech-control tags."
                    if self.language=="japanese"
                    else "Continue speaking in English using friendly tone and speech-control tags.")
        self.full_transcript.append({"role":"system","content":reminder})

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Voice Chat Assistant"
        }
        payload = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": self.full_transcript,
            "stream": True
        }

        print("Waiting for LLM stream...")
        try:
            full_reply = ""
            with httpx.stream("POST", "https://openrouter.ai/api/v1/chat/completions",
                              headers=headers, json=payload, timeout=60.0) as resp:
                for line in resp.iter_lines():
                    line = line.decode() if isinstance(line, bytes) else line
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        obj = json.loads(data)
                        delta = obj.get("choices", [{}])[0].get("delta", {}).get("content","")
                        if delta:
                            print(delta, end="", flush=True)
                            full_reply += delta
            if not full_reply.strip():
                full_reply = "I'm sorry, I didn't catch that. Could you say it again?"
            print()
            self.full_transcript.append({"role":"assistant","content":full_reply})
            self.generate_audio(full_reply)

        except Exception as e:
            print("LLM Error:", e)
            self.generate_audio("Sorry, I had a problem understanding. Please try again.")

    def generate_audio(self, text):
        print(f"\n🤖 Assistant (pre-TTS): {text}")
        clean = re.sub(r"<[^>]*>", "", text)
        try:
            wav = self.tts_model.generate(clean)
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.ndim==1:
                wav = wav.unsqueeze(0)
            audio = wav.squeeze().cpu().numpy()
            sd.play(audio, self.tts_model.sr)
            sd.wait()
            print("Finished speaking\n")
        except Exception as e:
            print("TTS Error:", e)

    def run(self):
        greeting = "<pitch level='high'>Hello!</pitch> <break> I'm your AI voice assistant. <rate level='slow'>What would you like to talk about today?</rate>"
        self.generate_audio(greeting)
        while True:
            audio = self.record_with_vad()
            if audio is not None:
                txt = self.transcribe(audio)
                if txt:
                    self.generate_ai_response(txt)

if __name__ == "__main__":
    AI_Assistant().run()
