# Stream-Voice-AI

An end-to-end, streaming voice assistant that listens to your speech, sends it through an LLM, and responds in natural-sounding voice—all within sub-2-second latency. Supports English & Japanese.

---

## Overview

- **Speech-to-Text (STT)**  
  - OpenAI Whisper (base/medium/large)  
  - WhisperX (with Voice Activity Detection)  

- **Large Language Models (LLM)**  
  - OpenRouter API → Google Gemma 12B/27B  
  - Mistral (via API)  

- **Text-to-Speech (TTS)**  
  - FishAudio S1 Mini (“Fish Speech”)  
  - CSA by CESMA  
  - Chatterbox TTS  

---

## Key Features

- **Ultra-low latency streaming**: As soon as you pause, you get a spoken reply.  
- **Multi-model pipeline**: STT, LLM, TTS model.  
- **Multi-language support**: English & Japanese .  
- **Modular architecture**: STT → LLM → TTS are separate services—mix & match.

---
## Methodology

1. **Voice Activity Detection**  
   - Continuously monitor the microphone stream.  
   - When audio amplitude exceeds threshold, start recording.  
   - When silence persists (VAD timeout), finalize the utterance.

2. **Streaming STT → LLM**  
   - Send each utterance chunk to the STT backend (WhisperX).  
   - Receive partial and final transcripts in real time.  
   - Forward transcript tokens immediately to the LLM service via OpenRouter or Mistral, enabling token-streaming.

3. **Generating the Response**  
   - LLM backend returns a token stream as it generates.  
   - Buffer or pass tokens directly to the TTS service for minimal delay.

4. **Streaming TTS Output**  
   - As each token arrives, call the chosen TTS endpoint (FishAudio, CSA, or Chatterbox) with streaming enabled.  
   - Pipe audio chunks back to the WebRTC client, so speech begins before full text generation.

5. **Multi-Model Swap-Out**  
   - Configuration flags in each service allow you to swap Whisper ↔ WhisperX, Gemma ↔ Mistral, or FishAudio ↔ CSA ↔ Chatterbox without touching the core pipeline code.

6. **Language Support**  
   - The frontend specifies `lang=en` or `lang=ja`.  
   - Backends load appropriate models/voice styles (e.g. Tsukasa for Japanese).


---

## Installation

1. Clone the repo  
   ```bash
   git clone https://github.com/lokesh9899/Stream-Voice-AI.git
   cd Stream-Voice-AI
   
2 Create & activate your venv
  python -m venv venv
  macOS/Linux
  source venv/bin/activate  
  Windows
  venv\Scripts\activate

3 Install dependencies
  pip install -r requirements.txt
