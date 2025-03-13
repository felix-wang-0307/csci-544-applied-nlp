from collections import defaultdict

def load_vocab(file_path):
    word2idx = defaultdict(lambda: word2idx["<UNK>"])  # Default to "<UNK>"
    word2idx["<PAD>"] = 0
    word2idx["<UNK>"] = 1

    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                _, word, _ = line.strip().split()
                if word not in word2idx:
                    word2idx[word] = len(word2idx)
    return word2idx

def load_tags():
    return defaultdict(lambda: 0, {
        'O': 0,
        'B-ORG': 1, 'I-ORG': 2,
        'B-PER': 3, 'I-PER': 4,
        'B-LOC': 5, 'I-LOC': 6,
        'B-MISC': 7, 'I-MISC': 8
    })


if __name__ == "__main__":
    word2idx = load_vocab("../data/train")
    tag2idx = load_tags()
    print(word2idx)
    print(tag2idx)