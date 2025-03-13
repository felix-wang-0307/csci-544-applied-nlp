import torch
from torch.utils.data import Dataset
import torch.nn as nn

class NERDataset(Dataset):
    def __init__(self, filepath, word2idx, tag2idx):
        self.sentences = []
        self.labels = []
        sentence, tags = [], []
        with open(filepath, 'r') as file:
            for line in file:
                if line.strip():
                    _, word, tag = line.strip().split()
                    sentence.append(word2idx.get(word, word2idx['<UNK>']))
                    tags.append(tag2idx[tag])
                else:
                    if sentence:
                        self.sentences.append(sentence)
                        self.labels.append(tags)
                        sentence, tags = [], []

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx]), torch.tensor(self.labels[idx])

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    xx_pad = nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=-1)
    return xx_pad, yy_pad, x_lens
