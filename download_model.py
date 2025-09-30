from huggingface_hub import snapshot_download
import os

# Try /work/model first (for Docker), then home directory
model_dir = "/work/model/whisper-large-v3"
if not os.path.exists("/work/model"):
    try:
        os.makedirs("/work/model", exist_ok=True)
    except:
        model_dir = os.path.expanduser("~/whisper-models/whisper-large-v3")
        os.makedirs(os.path.dirname(model_dir), exist_ok=True)

print(f"Downloading whisper-large-v3 model to {model_dir}...")
snapshot_download(
    repo_id="openai/whisper-large-v3",
    local_dir=model_dir
)
print(f"Download completed! Model saved to: {model_dir}")