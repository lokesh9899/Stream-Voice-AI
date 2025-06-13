# CSM-1B Transformers Streaming Audio Generator

This repository contains a Python implementation for generating streaming audio with the transformers implementation of CSM-1B (Contrastive Speech Model) model. It provides efficient, real-time audio generation with detailed performance metrics.

## Features

- 🔊 Real-time streaming audio generation
- ⚡ Optimized for low latency with chunk-based generation
- 📊 Detailed performance metrics (RTF - Real-Time Factor)
- 🎭 Support for reference audio samples

## Installation

### Requirements
* A CUDA-compatible GPU
* The code has been tested on CUDA 12.8 and 12.6, but it may also work on other versions
* Similarly, Python 3.10 is recommended, but newer versions may be fine
* For some audio operations, `ffmpeg` may be required

```bash
git clone git@github.com:davidbrowne17/csm-streaming-tf.git
cd csm-streaming-tf
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Basic Usage

### Loading the Model

```python
from generator import load_csm_1b

# Load the model from HuggingFace
generator = load_csm_1b("eustlb/csm-1b")
```

### Simple Text-to-Speech

```python
from generator import load_csm_1b, generate_streaming_audio

# Load the model
generator = load_csm_1b("eustlb/csm-1b")

# Generate audio from text
prompt = "The sunset painted the sky with hues of orange and purple."
audio = generate_streaming_audio(
    generator, 
    prompt, 
    output_filename="output.wav",
    play_audio=True
)
```

### Direct Streaming Generation

For more control over the generation process, you can use the streaming API directly:

```python
from generator import load_csm_1b, load_reference_audio
# Load the model from HuggingFace
generator = load_csm_1b("eustlb/csm-1b")
# Prepare reference audio data
reference_data = [
    {
        "path": "path/to/reference1.wav",
        "text": "This is a reference sample for voice cloning.",
        "speaker_id": "0"
    }
]

# Load the reference audio
refs = load_reference_audio(reference_data)

# Create conversation with reference audio
conversation = []
for ref in refs:
    conversation.append({
        "role": ref["speaker_id"],
        "content": [
            {"type": "text", "text": ref["text"]},
            {"type": "audio", "audio": ref["audio_array"]}
        ]
    })

# Add the current prompt
conversation.append({
    "role": refs[0]["speaker_id"],
    "content": [
        {"type": "text", "text": "Generate this text with the voice from the reference audio."}
    ]
})
inputs = generator.processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True
    ).to(generator.device)
# Stream generation with custom handling
for i, chunk in enumerate(generator.generate_stream(inputs, chunk_token_size=20)):
    # chunk is a tensor containing audio samples
    # process or play each chunk as needed
    print(f"Generated chunk {i+1} with {len(chunk.cpu().numpy()) / 24000:.3f} seconds of audio")
```

### Using Reference Audio

```python
from generator import load_csm_1b, load_reference_audio, generate_streaming_audio

# Load the model
generator = load_csm_1b("eustlb/csm-1b")

# Prepare reference audio data
reference_data = [
    {
        "path": "path/to/reference1.wav",
        "text": "This is a reference sample for voice cloning.",
        "speaker_id": "0"
    }
]

# Load the reference audio
refs = load_reference_audio(reference_data)

# Create conversation with reference audio
conversation = []
for ref in refs:
    conversation.append({
        "role": ref["speaker_id"],
        "content": [
            {"type": "text", "text": ref["text"]},
            {"type": "audio", "audio": ref["audio_array"]}
        ]
    })

# Add the current prompt
conversation.append({
    "role": refs[0]["speaker_id"],
    "content": [
        {"type": "text", "text": "Generate this text with the voice from the reference audio."}
    ]
})

# Generate audio
audio = generate_streaming_audio(
    generator,
    conversation,
    output_filename="with_reference.wav",
    play_audio=True
)
```

## API Reference

### Functions

#### `load_csm_1b(model_path="eustlb/csm-1b")`
Load the CSM-1B model from HuggingFace or a local path.

Parameters:
- `model_path`: Path to the model directory or HuggingFace model name

Returns:
- A fully initialized `Generator` instance

#### `generate_streaming_audio(generator, conversation, output_filename=None, play_audio=True, chunk_token_size=20, reference_data=None, **kwargs)`
Generate and play audio from a conversation, streaming chunks as they are generated.

Parameters:
- `generator`: The Generator instance to use
- `conversation`: Conversation history in the format expected by the processor, or a text prompt
- `output_filename`: Filename to save the generated audio to (optional)
- `play_audio`: Whether to play the audio in real time (default: True)
- `chunk_token_size`: Number of tokens to generate before yielding an audio chunk (default: 20)
- `reference_data`: Reference audio data to include in the conversation (optional)

Returns:
- The complete generated audio as a NumPy array

#### `load_reference_audio(reference_data)`
Load and process reference audio for the CSM model.

Parameters:
- `reference_data`: List of dictionaries containing reference data:
  - `path`: Path to the audio file
  - `text`: Text corresponding to the audio
  - `speaker_id`: Speaker ID for the audio

Returns:
- Processed reference data with audio arrays

### Generator Class

The main class for audio generation:

```python
generator = Generator(model, processor, device)
```

#### `generate_stream(inputs, chunk_token_size=20, **kwargs)`
Public method that streams audio chunks as they're generated.

Parameters:
- `inputs`: Processed inputs from the processor
- `chunk_token_size`: Number of tokens to generate before yielding an audio chunk (default: 20)
- `**kwargs`: Additional arguments to pass to the generator

Yields:
- Audio chunks as PyTorch tensors as they are generated

#### `_generate_stream(input_ids=None, input_values=None, input_values_cutoffs=None, generation_config=None, logits_processor=None, stopping_criteria=None, synced_gpus=None, chunk_token_size=20, **kwargs)`
Low-level method that handles the core streaming generation logic.

Parameters:
- `input_ids`: Tokenized input IDs
- `input_values`: Audio input values if applicable
- `input_values_cutoffs`: Cutoffs for audio inputs
- `generation_config`: Configuration for generation
- `logits_processor`: List of logits processors
- `stopping_criteria`: List of stopping criteria
- `synced_gpus`: Whether to sync across GPUs
- `chunk_token_size`: Number of codebook tokens to generate before yielding a chunk
- `**kwargs`: Additional arguments for generation

Yields:
- Raw audio chunks as PyTorch tensors in the range [-1, 1]

Details:
This method implements the core audio streaming functionality by:
1. Initializing RTF metrics tracking
2. Processing inputs and preparing the model
3. Running an initial forward pass
4. Entering the main generation loop:
   - Generate tokens for all codebooks
   - Check for EOS (end of sequence)
   - Accumulate tokens and yield audio chunks when `chunk_token_size` is reached
   - Decode audio using the codec model
   - Calculate and report RTF metrics
5. Providing comprehensive RTF metrics on completion

## Performance Metrics

The code automatically calculates and displays Real-Time Factor (RTF) metrics:

- RTF < 1.0: Generation is faster than real-time playback
- RTF = 1.0: Generation happens at exactly real-time
- RTF > 1.0: Generation is slower than real-time

The streaming implementation provides these metrics for each chunk and overall generation.

## FAQ

**How much faster is the streaming version?**

The perceived response time is significantly faster since you get the first audio chunks in milliseconds instead of waiting for the entire generation to complete. The actual total generation time is also improved by 40-60% depending on your hardware.

**Does this model come with any voices?**

The model is a base generation model capable of producing a variety of voices but hasn't been fine-tuned on any specific voice. Provide reference audio for best results.

**Can I converse with the model?**

CSM is trained to be an audio generation model and not a general-purpose multimodal LLM. It cannot generate text. Using a seperate LLM you can converse with the realtime demo via the web ui.

**Does it support other languages?**

The model has some capacity for non-English languages due to data contamination in the training data, but it likely won't do well.


## Misuse and abuse ⚠️

This project provides a high-quality speech generation model for research and educational purposes. While we encourage responsible and ethical use, we **explicitly prohibit** the following:

- **Impersonation or Fraud**: Do not use this model to generate speech that mimics real individuals without their explicit consent.
- **Misinformation or Deception**: Do not use this model to create deceptive or misleading content, such as fake news or fraudulent calls.
- **Illegal or Harmful Activities**: Do not use this model for any illegal, harmful, or malicious purposes.

By using this model, you agree to comply with all applicable laws and ethical guidelines. We are **not responsible** for any misuse, and we strongly condemn unethical applications of this technology.

---
## Acknowledgements

This code uses the CSM-1B model from [eustlb/csm-1b](https://huggingface.co/eustlb/csm-1b).

## Support me
Support this project on Ko-fi: https://ko-fi.com/davidbrowne17