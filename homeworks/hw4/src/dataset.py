import torch
from torch.utils.data import Dataset
import torch.nn as nn

class NERDataset(Dataset):
    def __init__(self, filepath, word2idx, tag2idx=None, has_tag=False):
        """
        Args:
            filepath: Path to dataset file.
            word2idx: Word to index mapping.
            tag2idx: Tag to index mapping (if available).
            is_test: Whether the dataset is test data (no tags available).
        """
        self.sentences = []
        self.labels = []
        sentence, tags = [], []

        with open(filepath, 'r') as file:
            for line in file:
                tokens = line.strip().split()
                
                if len(tokens) == 0:  # Sentence boundary
                    if sentence:
                        self.sentences.append(sentence)
                        if has_tag:  # Only add labels if it's not test data
                            self.labels.append(tags)
                        sentence, tags = [], []
                    continue

                if not has_tag:
                    # Test data: only index and word
                    _, word = tokens
                    sentence.append(word2idx.get(word, word2idx['<UNK>']))
                else:
                    # Train/Dev data: index, word, tag
                    _, word, tag = tokens
                    sentence.append(word2idx.get(word, word2idx['<UNK>']))
                    tags.append(tag2idx[tag])

        if sentence:
            self.sentences.append(sentence)
            if has_tag:
                self.labels.append(tags)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        if self.labels:  # Training/Dev mode
            return torch.tensor(self.sentences[idx]), torch.tensor(self.labels[idx])
        else:  # Test mode
            return torch.tensor(self.sentences[idx])

def pad_collate(batch):
    """
    Handles padding for variable-length sequences.
    Returns (xx_pad, yy_pad, x_lens) in all cases.
    """
    if isinstance(batch[0], tuple):  # Training/Dev Mode (with labels)
        (xx, yy) = zip(*batch)
        x_lens = [len(x) for x in xx]
        xx_pad = nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
        yy_pad = nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=-1)
        return xx_pad, yy_pad, x_lens
    else:  # Test Mode (no labels)
        xx = batch
        x_lens = [len(x) for x in xx]
        xx_pad = nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
        yy_pad = None  # Return None for labels in test mode
        return xx_pad, yy_pad, x_lens  # Consistent number of return values

