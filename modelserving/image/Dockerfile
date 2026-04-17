FROM docker.io/library/almalinux:latest

RUN dnf update -y

RUN dnf install -y python3.12-devel git

COPY imagesrv.ini /root/imagesrv.ini
COPY imagesrv.py /root/imagesrv.py
COPY requirements.txt /root/requirements.txt

WORKDIR /root

RUN python3 -m pip install --upgrade -r requirements.txt

CMD ["python3", "imagesrv.py"]
