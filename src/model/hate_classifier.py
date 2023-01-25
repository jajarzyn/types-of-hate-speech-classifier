import os
import time
from math import ceil
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments,\
    EarlyStoppingCallback, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from data_processing.data_reader import HateDataLoader, data_collator


class HateClassifier:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def restore_model(self, base_or_path, **model_kwargs):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_or_path,
            **model_kwargs,
        )

    def restore_tokenizer(self, base_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(base_or_path)

    def train(self, save_path, model_base_or_path, train_data_path, test_data_path, train_batch_size, eval_batch_size, num_epochs, early_stopping=None):
        print("Restoring tokenizer")
        self.restore_tokenizer(model_base_or_path)

        print("Reading train data")
        train_dataset = HateDataLoader(
            text_data_path=os.path.join(train_data_path, "training_set_clean_only_text.txt"),
            label_data_path=os.path.join(train_data_path, "training_set_clean_only_tags.txt"),
            balanced=True,
            tokenizer=self.tokenizer
        )

        print("Reading test data")
        test_dataset = HateDataLoader(
            text_data_path=os.path.join(test_data_path, "test_set_only_text.txt"),
            label_data_path=os.path.join(test_data_path, "test_set_only_tags.txt"),
            balanced=True,
            tokenizer=self.tokenizer
        )

        number_of_classes = len(set(train_dataset.labels))
        self.restore_model(model_base_or_path, num_labels=number_of_classes)

        training_args = TrainingArguments(
            output_dir=save_path,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            evaluation_strategy='epoch',
            logging_steps=50,
            logging_first_step=True,
            eval_steps=10,
            prediction_loss_only=False,
            learning_rate=8e-8,
            weight_decay=0.001,
            adam_epsilon=1e-8,
            num_train_epochs=num_epochs,
            save_strategy='epoch',
            warmup_ratio=0.1,
            dataloader_pin_memory=False,
            load_best_model_at_end=True,
            save_total_limit=10,
            fp16_full_eval=True,
            fp16=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping
                )
            ] if early_stopping else None
        )

        trainer.train()
        print(f'Saving model in: {save_path}')
        trainer.save_model()
        self.tokenizer.save_pretrained(save_path)

        test_dataset = HateDataLoader(
            text_data_path=os.path.join(test_data_path, "test_set_only_text.txt"),
            label_data_path=os.path.join(test_data_path, "test_set_only_tags.txt"),
            balanced=False,
            tokenizer=self.tokenizer
        )

        preds = trainer.predict(test_dataset)[0]
        labels = preds.argmax(axis=1)

        results_file = os.path.join(save_path, "results.txt")
        self.save_results(results_file, labels)

    def predict(
            self,
            save_path,
            text_data_path,
            tags_data_path,
            batch_size,
            model_path=None,
    ):
        if self.model is None and model_path is None:
            raise ValueError('Set model_base_or_path value if model is not loaded!')

        if self.tokenizer is None:
            self.restore_tokenizer(model_path)

        if self.model is None:
            self.restore_model(base_or_path=model_path)

        print("Reading test data")
        dataset = HateDataLoader(
            text_data_path=text_data_path,
            label_data_path=tags_data_path,
            balanced=False,
            tokenizer=self.tokenizer
        )

        device = 'cpu'  # as understood in task
        self.model.to(device)
        self.model.eval()

        start_time = time.time()

        with torch.no_grad():
            num_iters = ceil(len(dataset) / batch_size)

            preds = [torch.Tensor() for _ in range(num_iters)]
            for i in tqdm(range(num_iters), desc="Computing model predictions"):
                if i == num_iters - 1:
                    samples = (i * batch_size, len(dataset))

                else:
                    samples = (i * batch_size, (i + 1) * batch_size)

                preds[i] = self.model(**data_collator(
                    [dataset[j] for j in range(*samples)],
                    device=device
                )).logits

            preds = torch.concat(preds, dim=0)
            labels = torch.argmax(preds, dim=1)

        end_time = time.time()

        print(f"Execution time per sample: {(end_time - start_time) / len(dataset):.4f}s")

        self.save_results(os.path.join(save_path, 'results.txt'), labels)

    @staticmethod
    def save_results(save_path, labels):
        with open(save_path, "w") as results_file:
            for label in labels:
                results_file.write(f"{label}\n")

        print(f"Results saved in {save_path}")


def compute_metrics(pred_tuple):
    pred, labels = pred_tuple
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='macro')

    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1
    }
