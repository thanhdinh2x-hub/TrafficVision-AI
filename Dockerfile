FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip install opencv-python
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install pytesseract
RUN pip install ultralytics
RUN apt-get update && apt-get install -y tesseract-ocr

COPY ./runs /workspace/runs
COPY ./picture /workspace/picture
COPY yolo26n.pt .
COPY deploy.py .
