#!/bin/bash

echo Building docker image...

docker build -f Dockerfile -t mjarzyna/hate_clf .

echo Running container and prediction data...

docker container run \
-it mjarzyna/hate_clf python3 src/runner.py \
--action predict \
--save-path /types-of-hate-speech-classifier \
--data-path /types-of-hate-speech-classifier/data/test \
--model-path /types-of-hate-speech-classifier/res/model && \
echo Generating output.txt && \
perl /types-of-hate-speech-classifier/data/test/evaulate2.pl /types-of-hate-speech-classifier/results.txt > /types-of-hate-speech-classifier/output.txt && \
cat /types-of-hate-speech-classifier/output.txt

echo all done
