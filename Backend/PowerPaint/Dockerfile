FROM python:3.8.5-buster

WORKDIR /

COPY builder/requirements.txt .
RUN pip install -r requirements.txt

COPY builder/torchreqs.txt .
RUN pip install -r torchreqs.txt

ADD src .

CMD ["sh", "-c", "python -u /handler.py"]
