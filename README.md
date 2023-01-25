# types-of-hate-speech-classifier
PolEval2019 Task 6-2: Type of harmfulness.

# running model

## training

```bash
!python3 src/runner.py \
--action train \
--save-path MODEL_SAVE_PATH
```

## predict

```bash
python3 /content/types-of-hate-speech-classifier/src/runner.py \
--action predict \
--save-path SAVE_PATH \
--data-path DATA_DIR_PATH \
--model-path MODEL_DIR_PATH
```

# Docker

## Docker image building

```bash
docker build -f Dockerfile -t mjarzyna/hate_clf .
```

## Docker image saving

Docker image was saved and stored in repo to avoid needing of building it while checking "my homework".

```bash
docker save mjarzyna/hate_clf > hate_env.tar.gz
```

## Docker image run

Just run run.sh script

```bash
./run.sh
```
