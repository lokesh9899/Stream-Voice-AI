import torch
_real_torch_load = torch.load
def _torch_load_cpu(*args, **kwargs):
    if 'map_location' not in kwargs or kwargs['map_location'] is None:
        kwargs['map_location'] = torch.device('cpu')
    return _real_torch_load(*args, **kwargs)
torch.load = _torch_load_cpu

import os
import sys
import time
import re
import json
import tempfile

import numpy as np
import sounddevice as sd
import httpx
import whisper
from scipy.io.wavfile import write
from dotenv import load_dotenv
from chatterbox.tts import ChatterboxTTS

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY in .env")

SAMPLE_RATE     = 16000
RECORD_DURATION = 4 

class AI_Assistant:
    def __init__(self):
        self.full_transcript = [{
            "role": "system",
            "content": """
You are a friendly, natural-sounding AI assistant for voice conversations.

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
        self.last_prompt = None

        print("Loading Whisper (openai-whisper) medium.en model...")
        self.whisper_model = whisper.load_model("medium.en")
        print("Whisper model loaded")

        print("Loading ChatterboxTTS...")
        self.tts_model = ChatterboxTTS.from_pretrained(device="cpu")
        print("ChatterboxTTS loaded\n")

    def record_audio(self):
        """Record a fixed-duration clip from the microphone."""
        print(f"Listening for {RECORD_DURATION}s…")
        recording = sd.rec(int(RECORD_DURATION * SAMPLE_RATE),
                           samplerate=SAMPLE_RATE,
                           channels=1,
                           dtype="float32")
        sd.wait()
        return recording.squeeze(1) 

    def transcribe(self, audio_np):
        """
        Transcribe a NumPy array via Whisper.
        Returns the recognized text.
        """
        tmp_wav = os.path.join(tempfile.gettempdir(), "voice_temp.wav")
        write(tmp_wav, SAMPLE_RATE, (audio_np * 32767).astype("int16"))

        t0 = time.time()
        result = self.whisper_model.transcribe(tmp_wav, language="en")
        elapsed = time.time() - t0

        text = result.get("text", "").strip()
        if text:
            print(f"Transcribed in {elapsed:.1f}s: {text}")
        else:
            print(f"Transcribed in {elapsed:.1f}s: [no speech]")
        return text

    def generate_ai_response(self, user_input):
        """
        Send the user's input to the LLM via OpenRouter with streaming,
        then speak the full response via Chatterbox TTS.
        """
        print(f"\nUser: {user_input}\n")
        inp = user_input.lower()

        if inp in ["exit", "stop", "quit"]:
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
        reminder = (
            "From now on, reply only in Japanese using friendly tone and speech-control tags."
            if self.language == "japanese"
            else "Continue speaking in English using friendly tone and speech-control tags."
        )
        self.full_transcript.append({"role":"system","content":reminder})

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Voice Chat Assistant",
        }
        payload = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": self.full_transcript,
            "stream": True
        }

        print("Waiting for LLM stream…")
        try:
            full_reply = ""
            with httpx.stream("POST",
                              "https://openrouter.ai/api/v1/chat/completions",
                              headers=headers,
                              json=payload,
                              timeout=60.0) as resp:
                for line in resp.iter_lines():
                    line = line.decode() if isinstance(line, bytes) else line
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        print(delta, end="", flush=True)
                        full_reply += delta
            if not full_reply.strip():
                full_reply = "I'm sorry, I didn't catch that. Here's an answer anyway."
            print() 
            self.full_transcript.append({"role":"assistant","content":full_reply})
            self.generate_audio(full_reply)

        except Exception as e:
            print("LLM Error:", e)
            self.generate_audio("Sorry, something went wrong. Please try again.")

    def generate_audio(self, text):
        """
        Speak the given text via ChatterboxTTS.
        Strips SSML/prosody tags before synthesis.
        """
        print(f"\n🤖 Assistant (pre-TTS): {text}")
        clean = re.sub(r"<[^>]+>", "", text)
        try:
            wav = self.tts_model.generate(clean)
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            audio = wav.squeeze().cpu().numpy()
            sd.play(audio, self.tts_model.sr)
            sd.wait()
            print("Finished speaking\n")
        except Exception as e:
            print("TTS Error:", e)

    def run(self):
        """Main interaction loop."""
        greeting = ("<pitch level='high'>Hello!</pitch> <break> I'm your AI voice assistant. "
                    "<rate level='slow'>What would you like to talk about today?</rate>")
        self.generate_audio(greeting)

        while True:
            audio = self.record_audio()
            text  = self.transcribe(audio)
            if text:
                self.last_prompt = text
                self.generate_ai_response(text)
            elif self.last_prompt:
                print("Didn’t catch that—repeating last question to LLM.")
                self.generate_ai_response(self.last_prompt)
            else:
                self.generate_audio("Sorry, I didn't catch that. Please try again.")

if __name__ == "__main__":
    AI_Assistant().run()
