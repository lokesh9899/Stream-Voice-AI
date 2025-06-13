import os
import time
import whisperx
import httpx
from dotenv import load_dotenv
from scipy.io.wavfile import write
from datetime import datetime
import torch
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from pydub.playback import play

from generator import load_csm_1b, generate_streaming_audio

SAFE_TTS_DIR = r'C:\Users\lokes\OneDrive\Documents\streaming-voice\tmp'
os.makedirs(SAFE_TTS_DIR, exist_ok=True)
os.makedirs("audio", exist_ok=True)

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY in .env")

SAMPLE_RATE = 16000
CHANNELS = 1
VOLUME_THRESHOLD = 0.02
MAX_DURATION = 10
SILENCE_TIMEOUT = 0.5
MIN_SPEECH_DURATION = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔄 Loading WhisperX tiny.en model on {device}...")
model = whisperx.load_model(
    "tiny.en", device=device,
    compute_type=("float16" if device == "cuda" else "int8")
)
print("WhisperX model loaded")

print("Loading CSM-1B TTS model (this may take a minute)...")
tts_generator = load_csm_1b("eustlb/csm-1b")
print("CSM-1B TTS loaded")


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


def speak_with_csm_tts(text, output_path=None, play_audio=True):
    if not text.strip():
        print("No text to speak.")
        return
    output_path = output_path or os.path.join(SAFE_TTS_DIR, f"tts_{datetime.now():%Y%m%d_%H%M%S}.wav")
    print("🗣️ Generating speech with CSM-1B TTS...")
    try:
        generate_streaming_audio(
            tts_generator,
            text,
            output_filename=output_path,
            play_audio=False
        )
        print(f"Audio saved: {output_path}")
        if play_audio:
            audio=AudioSegment.from_file(output_path, format="wav")
            play(audio)
        # If not played by generate_streaming_audio, you can play with pydub:
        # audio = AudioSegment.from_file(output_path, format="wav")
        # play(audio)
    except Exception as e:
        print(f"TTS Error: {e}")


def chat_and_speak(prompt, language="english"):
    sys_en = (
        "You are Lexi, a friendly AI voice assistant. Respond with plain, natural, and conversational English text only. "
        "Do not use SSML tags or special markup. Speak as if you're talking to a friend in a warm, approachable tone."
    )
    sys_jp = (
        "あなたはLexiという親しみやすい音声アシスタントです。返答は日本語の自然な会話調テキストだけにしてください。タグや特別な記法は不要です。"
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
        "messages": [sys_msg, {"role": "user", "content": prompt}]
    }

    print("🧠 Waiting for full LLM response...")
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, headers=headers, json=data)
            if response.status_code != 200:
                print("LLM Error:", response.status_code)
                print("Raw response:", response.text)
                return
            try:
                j = response.json()
            except Exception as e:
                print("LLM JSON parse error:", e)
                print("Raw response:", response.text)
                return
            content = ""
            try:
                content = j["choices"][0]["message"]["content"]
            except Exception as e:
                print("LLM: Response JSON format unexpected:", e)
                print("LLM JSON:", j)
                return

            print(f"\n🤖 Response:\n  {content}")
            speak_with_csm_tts(content, play_audio=True)
    except Exception as ex:
        print("HTTP or API error:", ex)
        return


if __name__ == '__main__':
    print("Voice Assistant Started (EN/JP, plain-text mode, CSM-1B TTS, pydub playback)")
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
