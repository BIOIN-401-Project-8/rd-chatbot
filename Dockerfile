FROM nvcr.io/nvidia/pytorch:23.11-py3

RUN apt update && apt install -y graphviz && apt clean

# Install Entrez Direct
# https://www.ncbi.nlm.nih.gov/books/NBK179288/
RUN yes | sh -c "$(curl -fsSL https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)"

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
