from huggingface_hub import snapshot_download
import os

model_dir = os.path.expanduser("~/whisper-models/whisper-large-v3")
print(f"Downloading whisper-large-v3 model to {model_dir}...")
snapshot_download(
    repo_id="openai/whisper-large-v3",
    local_dir=model_dir
)
print(f"Download completed! Model saved to: {model_dir}")