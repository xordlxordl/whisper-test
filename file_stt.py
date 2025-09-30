import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import sys

# Setup device and dtype
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model path
model_id = "/work/model/whisper-large-v3"
if not os.path.exists(model_id):
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

print(f"Model loaded successfully on {device}!")
print("\n" + "="*50)
print("Audio File Speech-to-Text")
print("="*50 + "\n")

if len(sys.argv) < 2:
    print("Usage: python file_stt.py <audio_file>")
    print("Example: python file_stt.py audio.wav")
    sys.exit(1)

audio_file = sys.argv[1]

if not os.path.exists(audio_file):
    print(f"Error: File '{audio_file}' not found!")
    sys.exit(1)

print(f"Processing: {audio_file}")
print("Transcribing...")

try:
    result = pipe(audio_file)

    print("\n" + "-"*50)
    print("Transcription:")
    print(result["text"])
    print("-"*50 + "\n")

except Exception as e:
    print(f"Error occurred: {e}")
    sys.exit(1)