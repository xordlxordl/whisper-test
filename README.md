# Make Docker Image
- docker build -t whisper-model-image .

# Download model 
- python download_model.py

## 방법 1: 오디오 장치 마운트 후 재실행

 docker run -it --gpus all --ipc=host --device /dev/snd --group-add audio -e AUDIODEV=hw:0,0 whisper-model-image bash
 python mic_stt.py


## 방법 2: 파일 기반 STT 사용

  file_stt.py. 호스트에서 오디오 파일을 녹음하고 Docker로 전달:

  ### 호스트에서 오디오 녹음
  arecord -d 5 -f cd -t wav audio.wav

  ### Docker 컨테이너에 파일 복사
  docker cp audio.wav <container_id>:/workspace/

  ### 컨테이너 내부에서 실행
  python file_stt.py audio.wav


## 방법 3: 현재 컨테이너에서 수정

  컨테이너 내부에서:
  apt-get update
  apt-get install -y portaudio19-dev alsa-utils
  pip3 install sounddevice --force-reinstall

  ### 오디오 장치 확인
  arecord -l

  ### 실행
  python mic_stt.py