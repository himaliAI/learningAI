# NLP Foundation with TorchText (IMDB Sentiment Dataset)

import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator

# load dataset
train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

# one example
label, text = next(iter(train_iter))

# Tokenize the text (words or subwords)
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer(text)

# build a vocabulary
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# reload train_iter as it was comsumed above
train_iter = IMDB(split='train')

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

print(f"{len(vocab)}") # size of vocabulary
print(vocab(["this", "movie", "is", "great"]))

# Numericalization Pipeline
def text_pipeline(x):
    return vocab(tokenizer(x))

def label_pipeline(y):
    return 1 if y == 'pos' else 0

# collate function for batching
def collate_batch(batch):
    label_list, text_list = [], []
    for label, text in batch:
        label_list.append(label_pipeline(label))
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text_list.append(processed_text)
    labels = torch.tensor(label_list, dtype=torch.int64)
    texts = pad_sequence(text_list, batch_first=True)
    return labels, texts

# DataLoader with Batching
train_iter = IMDB(split='train') # reload again for DataLoader
dataloader = DataLoader(train_iter, batch_size=8, collate_fn=collate_batch)

for labels, texts in dataloader:
    print(f"Batch labels shape: {labels.shape}")
    print(f"Batch texts shape: {texts.shape}")
    print(f"First batch labels: {labels}")
    print(f"First batch texts (first row): {texts[0][:20]}")
    break

# This script:
    # Loads IMDB dataset (train/test)
    # Tokenizes text using basic_english
    # Builds a vocabulary from training data
    # Defines pipelines for text -> IDs and label ->0/1
    # Pads sequences to equal length in batches
    # Creates a DataLoader that yields batches of (labels, texts) tensors