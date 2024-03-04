FROM nvcr.io/nvidia/pytorch:23.11-py3

RUN apt update && apt install -y graphviz && apt clean

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
