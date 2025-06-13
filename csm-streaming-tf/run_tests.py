import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import queue
import threading
from generator import (
    load_csm_1b,
    generate_streaming_audio,
    load_reference_audio
)

def test_stream_generation(generator, prompt):
    """Test the streaming generation method directly"""
    print("\n" + "="*50)
    print("Testing stream generation directly...")
    print("="*50)
    
    # Prepare input
    inputs = generator.processor.apply_chat_template(
        [{
            "role": "0",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }],
        tokenize=True,
        return_dict=True
    ).to(generator.device)
    
    # Time the generation
    start_time = time.time()
    
    # Collect chunks
    chunks = []
    for i, chunk in enumerate(generator.generate_stream(inputs, chunk_token_size=20)):
        chunk_numpy = chunk.cpu().numpy().astype(np.float32)  # Ensure float32 type
        chunks.append(chunk_numpy)
        print(f"Received chunk {i+1} with {len(chunk_numpy) / 24000:.3f} seconds of audio")
    
    generation_time = time.time() - start_time
    
    # Concatenate chunks
    if chunks:
        audio = np.concatenate(chunks)
        duration = len(audio) / 24000
        rtf = generation_time / duration if duration > 0 else 0
        
        print(f"Generated {duration:.2f} seconds of audio in {generation_time:.2f} seconds (RTF: {rtf:.4f})")
        
        # Save the audio
        output_file = "stream_generation.wav"
        sf.write(output_file, audio, 24000)
        print(f"Saved audio to {output_file}")
        
        return audio
    else:
        print("No audio chunks were generated")
        return None

def test_streaming_audio_helper(generator, prompt):
    """Test the streaming audio helper function"""
    print("\n" + "="*50)
    print("Testing streaming audio helper function...")
    print("="*50)
    
    output_file = "streaming_audio_helper.wav"
    
    # Time the generation
    start_time = time.time()
    
    # Generate audio with helper function
    audio = generate_streaming_audio(
        generator,
        prompt,
        output_filename=output_file,
        play_audio=True,  # Real-time playback
        chunk_token_size=20
    )
    
    generation_time = time.time() - start_time
    
    if audio is not None:
        duration = len(audio) / 24000
        rtf = generation_time / duration if duration > 0 else 0
        
        print(f"Generated {duration:.2f} seconds of audio in {generation_time:.2f} seconds (RTF: {rtf:.4f})")
        
        return audio
    else:
        print("No audio was generated")
        return None

def test_with_reference_audio(generator, reference_data, prompt, audio_queue=None):
    """Test generation with reference audio for voice cloning"""
    print("\n" + "="*50)
    print("Testing generation with reference audio...")
    print("="*50)
    
    # Create conversation with reference audio context
    base_conversation = []
    for ref in reference_data:
        base_conversation.append({
            "role": ref["speaker_id"],
            "content": [
                {"type": "text", "text": ref["text"]},
                {"type": "audio", "audio": ref["audio_array"]}
            ]
        })
    
    # Add the current prompt for generation
    final_conversation = base_conversation.copy()
    final_conversation.append({
        "role": reference_data[0]["speaker_id"],
        "content": [
            {"type": "text", "text": prompt}
        ]
    })
    
    # Process inputs
    inputs = generator.processor.apply_chat_template(
        final_conversation,
        tokenize=True,
        return_dict=True
    ).to(generator.device)
    
    # Output filename
    output_file = "with_reference_audio.wav"
    
    # Stream generation
    chunks = []
    
    # Open WAV file for writing
    with sf.SoundFile(output_file, "w", samplerate=24000, channels=1, subtype="PCM_16") as wav_fh:
        # Stream generation and queue chunks as they arrive
        for pcm_chunk in generator.generate_stream(inputs, chunk_token_size=20):
            # Convert to numpy
            np_chunk = pcm_chunk.cpu().numpy()
            chunks.append(np_chunk)
            
            # Add to playback queue if provided
            if audio_queue is not None:
                audio_queue.put(np_chunk)
            
            # Save to file
            wav_fh.write(np_chunk)
            
            # Print progress indicator
            print(".", end="", flush=True)
    
    print(f"\nSaved audio to {output_file}")
    
    # Concatenate chunks
    if chunks:
        audio = np.concatenate(chunks)
        duration = len(audio) / 24000
        print(f"Generated {duration:.2f} seconds of audio with reference voice")
        
        return audio
    else:
        print("No audio was generated")
        return None

def generate_streaming_audio_with_reference(generator, reference_data, prompt, output_filename=None, 
                                           play_audio=False, chunk_token_size=20):
    """Helper function for generating streaming audio with reference data for voice cloning"""
    # Create conversation with reference audio context
    base_conversation = []
    for ref in reference_data:
        base_conversation.append({
            "role": ref["speaker_id"],
            "content": [
                {"type": "text", "text": ref["text"]},
                {"type": "audio", "audio": ref["audio_array"]}
            ]
        })
    
    # Add the current prompt for generation
    final_conversation = base_conversation.copy()
    final_conversation.append({
        "role": reference_data[0]["speaker_id"],
        "content": [
            {"type": "text", "text": prompt}
        ]
    })
    
    # Process inputs
    inputs = generator.processor.apply_chat_template(
        final_conversation,
        tokenize=True,
        return_dict=True
    ).to(generator.device)
    
    # Create audio queue if needed for real-time playback
    audio_queue = None
    playback_thread = None
    if play_audio:
        audio_queue = queue.Queue()
        
        def audio_playback_thread():
            while True:
                chunk = audio_queue.get()
                if chunk is None:  # None is our signal to stop
                    break
                sd.play(chunk.astype("float32"), 24000)
                sd.wait()
                audio_queue.task_done()
        
        playback_thread = threading.Thread(target=audio_playback_thread)
        playback_thread.daemon = True
        playback_thread.start()
    
    # Stream generation with chunk collection
    chunks = []
    
    # Handle file output if needed
    if output_filename:
        wav_file = sf.SoundFile(output_filename, "w", samplerate=24000, channels=1, subtype="PCM_16")
    else:
        wav_file = None
    
    try:
        # Stream generation and process chunks as they arrive
        for pcm_chunk in generator.generate_stream(inputs, chunk_token_size=chunk_token_size):
            # Convert to numpy
            np_chunk = pcm_chunk.cpu().numpy()
            chunks.append(np_chunk)
            
            # Add to playback queue if requested
            if audio_queue is not None:
                audio_queue.put(np_chunk)
            
            # Save to file if requested
            if wav_file is not None:
                wav_file.write(np_chunk)
            
            # Print progress indicator
            print(".", end="", flush=True)
    
    finally:
        # Clean up resources
        if wav_file is not None:
            wav_file.close()
            print(f"\nSaved audio to {output_filename}")
        
        # Stop playback thread if it exists
        if playback_thread is not None:
            audio_queue.put(None)
            playback_thread.join()
    
    # Concatenate chunks
    if chunks:
        audio = np.concatenate(chunks)
        return audio
    else:
        return None

def main():
    """Main test function"""
    # Create results directory
    os.makedirs("test_results", exist_ok=True)
    os.chdir("test_results")
    
    print("Starting CSM Generator tests...")
    
    # Test prompts
    test_prompt = "The sunset painted the sky with hues of orange and purple, creating a stunning backdrop against the mountain range."
    
    # Define reference data
    reference_data = [
        {
            "path": "path_to_reference.wav",
            "text": "the reference text of your wav file",
            "speaker_id": "0"
        }
    ]
    
    # Create audio queue and playback thread for real-time audio
    audio_queue = queue.Queue()
    
    def audio_playback_thread():
        while True:
            chunk = audio_queue.get()
            if chunk is None:  # None is our signal to stop
                break
            sd.play(chunk.astype("float32"), 24000)
            sd.wait()  # Wait here is fine as it's in a separate thread
            audio_queue.task_done()
    
    # Start playback thread
    playback_thread = threading.Thread(target=audio_playback_thread)
    playback_thread.daemon = True
    playback_thread.start()
    
    try:
        # Step 1: Load generator directly from HuggingFace
        generator = load_csm_1b("eustlb/csm-1b")
        
        # Step 2: Load references
        refs = load_reference_audio(reference_data)
        
        # Step 3: Test stream generation
        stream_audio = test_stream_generation(generator, test_prompt)
        
        # Step 4: Test streaming audio helper
        helper_audio = test_streaming_audio_helper(generator, test_prompt)
        
        # Step 5: Test with reference audio
        ref_audio = test_with_reference_audio(generator, refs, test_prompt, audio_queue)
                
        # Wait for all audio to finish playing
        audio_queue.join()
        
        # Signal playback thread to stop
        audio_queue.put(None)
        playback_thread.join()
        
        print("\n" + "="*50)
        print("All tests completed successfully!")
        print("="*50)
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Make sure we always try to stop the playback thread
        if playback_thread.is_alive():
            audio_queue.put(None)

if __name__ == "__main__":
    main()