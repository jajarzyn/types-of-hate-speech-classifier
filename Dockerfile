FROM python:3.11.1-slim

COPY . /types-of-hate-speech-classifier

WORKDIR /types-of-hate-speech-classifier

RUN python -m pip install -r requirements.txt

ENV PYTHONPATH=PYTHONPATH:/types-of-hate-speech-classifier/src

# Docker check-up:
#
#RUN python3 src/runner.py \
#--action predict \
#--save-path /types-of-hate-speech-classifier \
#--data-path /types-of-hate-speech-classifier/data/test \
#--model-path /types-of-hate-speech-classifier/res/model
#
#RUN perl /types-of-hate-speech-classifier/data/test/evaulate2.pl \
#/types-of-hate-speech-classifier/results.txt> \
#/types-of-hate-speech-classifier/output.txt
#
#RUN cat /types-of-hate-speech-classifier/output.txt
