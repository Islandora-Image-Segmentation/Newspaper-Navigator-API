FROM python:3.7.7

RUN apt update -y
RUN apt upgrade -y 

RUN apt install -y tesseract-ocr
RUN apt install -y libtesseract-dev

RUN pip install -U 'git+https://github.com/facebookresearch/fvcore'
RUN pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY ./src /src

ENTRYPOINT ["python", "/src/main.py"]