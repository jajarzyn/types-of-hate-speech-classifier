import argparse
import os
from model.hate_classifier import HateClassifier


def main(args):
    current_path = os.path.dirname(os.path.abspath(__file__))
    classifier = HateClassifier()

    if args.action == 'train':

        classifier.train(
            save_path=args.save_path,
            model_base_or_path="allegro/herbert-base-cased",
            train_data_path=os.path.join(current_path, "../data/train"),
            test_data_path=os.path.join(current_path, "../data/test"),
            train_batch_size=128,
            eval_batch_size=256,
            num_epochs=15,
            early_stopping=5
        )

    else:
        if args.data_path is None:
            raise ValueError("Pass --data-path argument for prediction!")

        text_file, tags_file = get_data_files(args.data_path)
        classifier.predict(
            save_path=args.save_path,
            text_data_path=os.path.join(args.data_path, text_file),
            tags_data_path=os.path.join(args.data_path, tags_file),
            batch_size=256,
            model_path=os.path.join(current_path, "../res/model") if args.model_path is None else args.model_path,
        )


def get_data_files(path):
    files = os.listdir(path)

    text_files = [file for file in files if "text" in file]
    tags_files = [file for file in files if "tags" in file]

    assert len(text_files) == 1 and len(tags_files) == 1, f"Invalid data directory content: {files}"

    return text_files[0], tags_files[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--action", type=str, choices=['train', 'predict'])
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--model-path")
    parser.add_argument('--save-path', type=str, required=True)

    args = parser.parse_args()
    main(args)
