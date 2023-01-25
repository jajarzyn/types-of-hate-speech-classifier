FROM python:3.11.1-slim

#RUN apt-get install curl -y && \
#    curl -O https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh \
#    && bash script.deb.sh

RUN mkdir /types-of-hate-speech-classifier

COPY src /types-of-hate-speech-classifier/src
COPY data /types-of-hate-speech-classifier/data
COPY res /types-of-hate-speech-classifier/res
COPY requirements.txt /types-of-hate-speech-classifier/requirements.txt

RUN ls -l

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
