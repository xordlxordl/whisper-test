# Make Docker Image
- docker build -t whisper-model-image .
- docker run whisper-model-image

# Execute 
- docker run -it whisper-model-image bash
- python download_model.py
- python mic_stt.py