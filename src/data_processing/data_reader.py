from collections import Counter
from itertools import chain
from random import sample, shuffle
from tqdm import tqdm
import torch


class HateDataLoader:
    def __init__(
            self,
            text_data_path,
            label_data_path,
            tokenizer,
            max_length=128,
            balanced=False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.balanced = balanced

        self.labels = None

        num_lines_text = sum(1 for _ in open(text_data_path))
        if label_data_path is not None:
            num_lines_labels = sum(1 for _ in open(label_data_path))

        assert num_lines_text == num_lines_labels,\
            "Invalid input files, number of texts and labels must be equal."

        self.texts = ["" for _ in range(num_lines_text)]
        if label_data_path is not None:
            self.labels = [0 for _ in range(num_lines_labels)]

        with open(text_data_path, 'r') as text_file:
            for i, line in enumerate(text_file):
                self.texts[i] = line.replace('\n', '')

        if label_data_path is not None:
            with open(label_data_path, 'r') as label_file:
                for i, line in enumerate(label_file):
                    self.labels[i] = int(line.replace('\n', ''))

        self.texts_tokenized = self.tokenize_texts()

        self.indexer, self.index_queue, self.to_proc = self.get_indexer()

    def __len__(self):
        return len(self.indexer)

    def __getitem__(self, item):
        if not self.index_queue[item]:
            self.index_queue[item] = True
            self.to_proc -= 1

        idx = self.indexer[item]

        if self.to_proc <= 0 and all(self.index_queue):
            self.indexer, self.index_queue, self.to_proc = self.get_indexer()

        encoding = self.texts_tokenized[idx]
        if self.labels is not None:
            encoding['labels'] = torch.LongTensor([self.labels[idx]])

        return encoding

    def get_indexer(self):
        if self.balanced and self.labels is not None:
            return self.oversample()

        else:
            indexer = {i: i for i in range(len(self.texts))}
            index_queue = [False] * len(indexer)
            to_proc = len(indexer)
            return indexer, index_queue, to_proc

    def oversample(self):
        labels_counter = Counter(self.labels)

        labels_count = labels_counter.most_common()
        number_of_texts_to_sample = labels_count[0][1]
        oversampled_indexes = {}

        for label, label_count in labels_count:
            label_lines = [i for i, _label in enumerate(self.labels) if _label == label]

            if len(label_lines) < number_of_texts_to_sample:
                label_lines = label_lines * (number_of_texts_to_sample // len(label_lines) + 1)
                oversampled_indexes[label] = sample(label_lines, number_of_texts_to_sample)

            else:
                oversampled_indexes[label] = label_lines

        return self.create_indexer(oversampled_indexes)

    def create_indexer(self, oversampled_indexes):

        oversampled_idx = [i for i in chain.from_iterable(
            [indexes for indexes in oversampled_indexes.values()]
        )]
        shuffle(oversampled_idx)

        dataloader_idx = [i for i in range(len(self.labels))]

        indexer = {
            idx: data_idx for idx, data_idx in zip(dataloader_idx, oversampled_idx)
        }
        index_queue = [False] * len(indexer)
        to_proc = len(indexer)

        return indexer, index_queue, to_proc

    def tokenize_texts(self):
        return [
            self.tokenizer(
                text,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt',
            ) for text in tqdm(self.texts, desc="Tokenizing texts")
        ]


def data_collator(encodings, device=None):
    batch = {
        'input_ids': torch.stack(
            [encoding['input_ids'][0] for encoding in encodings]
        ),
        'attention_mask': torch.stack(
            [encoding['attention_mask'][0] for encoding in encodings]
        ),
    }

    if hasattr(encodings[0], 'labels'):
        batch['labels'] = torch.stack(
            [encoding['labels'] for encoding in encodings]
        )
    if device is not None:
        batch = {key: tensor.to(device) for key, tensor in batch.items()}

    return batch
