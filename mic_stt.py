import torch
import sounddevice as sd
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os

# Setup device and dtype
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model path (update this to match where you downloaded the model)
model_id = os.path.expanduser("~/whisper-models/whisper-large-v3")

print("Loading model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

print("Model loaded successfully!")

# Check available audio devices
print("\nAvailable audio devices:")
print(sd.query_devices())

# Get default input device
try:
    default_device = sd.default.device[0]
    device_info = sd.query_devices(default_device, 'input')
    print(f"\nUsing device: {device_info['name']}")
    print(f"Default sample rate: {device_info['default_samplerate']}")
except Exception as e:
    print(f"\nWarning: Could not detect default device: {e}")
    print("Attempting to use first available input device...")
    devices = sd.query_devices()
    input_devices = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    if input_devices:
        default_device = input_devices[0]
        device_info = sd.query_devices(default_device, 'input')
        sd.default.device = default_device
        print(f"Using device: {device_info['name']}")
    else:
        print("ERROR: No input devices found!")
        exit(1)

print("\n" + "="*50)
print("Recording Settings:")
print("  - Sample Rate: 16000 Hz")
print("  - Duration: 5 seconds")
print("  - Press Ctrl+C to exit")
print("="*50 + "\n")

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone"""
    print(f"Recording for {duration} seconds...")
    try:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        print("Recording finished!")
        return audio.squeeze()
    except Exception as e:
        print(f"Recording error: {e}")
        raise

# Main loop
try:
    while True:
        # Record audio from microphone
        audio = record_audio(duration=5)

        # Convert to the format expected by the pipeline
        audio_input = {"array": audio, "sampling_rate": 16000}

        # Perform speech-to-text
        print("Transcribing...")
        result = pipe(audio_input)

        # Print result
        print("\n" + "-"*50)
        print("Transcription:")
        print(result["text"])
        print("-"*50 + "\n")

        # Ask if user wants to continue
        cont = input("Press Enter to record again, or 'q' to quit: ")
        if cont.lower() == 'q':
            break

except KeyboardInterrupt:
    print("\n\nExiting...")
except Exception as e:
    print(f"\nError occurred: {e}")