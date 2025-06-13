import os
import json
import time
import asyncio
import sounddevice as sd
import numpy as np
import whisperx
import httpx
from dotenv import load_dotenv
from scipy.io.wavfile import write
from datetime import datetime
import torch
from fish_audio_sdk import Session, TTSRequest

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
FISHAUDIO_API_KEY = os.getenv("FISHAUDIO_API_KEY")
if not OPENROUTER_API_KEY or not FISHAUDIO_API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY or FISHAUDIO_API_KEY in .env")

SAMPLE_RATE = 16000
CHANNELS = 1
TTS_SAMPLE_RATE = 22050
VOLUME_THRESHOLD = 0.02
MAX_DURATION = 10
SILENCE_TIMEOUT = 0.5
MIN_SPEECH_DURATION = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading WhisperX tiny.en model on {device}...")
model = whisperx.load_model(
    "tiny.en", device=device,
    compute_type=("float16" if device == "cuda" else "int8")
)
print("WhisperX model loaded")

ws_tts = Session(FISHAUDIO_API_KEY)

def record_with_vad():
    print("Listening (VAD enabled)... Speak now.")
    frames, total, silence = [], 0, 0
    silence_frames = int(SAMPLE_RATE * SILENCE_TIMEOUT)
    min_frames = int(SAMPLE_RATE * MIN_SPEECH_DURATION)

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

    audio_data = np.concatenate(frames, axis=0)
    os.makedirs("audio", exist_ok=True)
    filename = f"audio/{datetime.now():%Y%m%d_%H%M%S}.wav"
    write(filename, SAMPLE_RATE, (audio_data * 32767).astype("int16"))
    return filename

def transcribe(path):
    audio = whisperx.load_audio(path)
    t0 = time.time()
    res = model.transcribe(audio, batch_size=8, language="en")
    elapsed = time.time() - t0
    text = " ".join(s["text"] for s in res.get("segments", []))
    print(f"Transcribed in {elapsed:.1f}s: {text}")
    return text.strip()

async def _play_ws(text, use_ssml=True):
    if use_ssml:
        text = f"<speak><prosody rate='1.05' pitch='+8%'>{text}</prosody></speak>"
    try:
        req = TTSRequest(
            text=text,
            backend="s1-mini", 
            format="pcm"
        )
        with sd.RawOutputStream(samplerate=TTS_SAMPLE_RATE, channels=1, dtype="int16") as stream:
            print("Speaking...")
            for chunk in ws_tts.tts(req):
                if isinstance(chunk, bytes):
                    stream.write(chunk)
        print("Finished speaking.")
    except Exception as e:
        print(f"TTS Error: {e}")

def speak_stream(text, use_ssml=True):
    try:
        asyncio.run(_play_ws(text, use_ssml))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_play_ws(text, use_ssml))

def chat_and_speak(prompt, language="english"):
    sys_en = (
        "You are Lexi, a friendly voice assistant. "
        "Use tags like <break>, <pitch>, <rate> for prosody. "
        "Example: <pitch level='high'>Hi!</pitch> <break time='200ms'>"
    )
    sys_jp = (
        "Lexiへようこそ。フレンドリーな音声アシスタントです。"
        " タグ (<break>,<rate>,<pitch>,<emphasis>) を使って自然な話し方にしてください。"
    )
    sys_msg = {'role': 'system', 'content': sys_jp if language == 'japanese' else sys_en}

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "VoiceChat",
        "Content-Type": "application/json"
    }
    data = {
        "model": os.getenv("MODEL_NAME", "mistralai/mistral-7b-instruct"),
        "stream": False,
        "messages": [sys_msg, {'role': 'user', 'content': prompt}]
    }

    print("Waiting for full LLM response...")
    try:
        resp = httpx.post(url, headers=headers, json=data, timeout=60.0)
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
        print("\n🤖 Response:\n", reply)
        speak_stream(reply)
        print("\nDone\n")
    except Exception as e:
        print(f"LLM Error: {e}")

# Run loop
if __name__ == '__main__':
    print("Voice Assistant Started (EN/JP, full-response mode)")
    while True:
        try:
            wav = record_with_vad()
            if wav:
                txt = transcribe(wav)
                if txt:
                    lang = 'japanese' if any(k in txt for k in ['日本語', 'にほんご']) else 'english'
                    chat_and_speak(txt, language=lang)
        except KeyboardInterrupt:
            print("Exiting.")
            break
