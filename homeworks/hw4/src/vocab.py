from collections import defaultdict

def generate_vocab(file_path, output_path="./out/vocab.txt"):
    word2idx = defaultdict(lambda: word2idx["<UNK>"])  # Default to "<UNK>"
    word2idx["<PAD>"] = 0
    word2idx["<UNK>"] = 1

    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                _, word, _ = line.strip().split()
                if word not in word2idx:
                    word2idx[word] = len(word2idx)
    
    if output_path:
        with open(output_path, "w") as f:
            for word, idx in word2idx.items():
                f.write(f"{idx} {word}\n")
    
    return word2idx

def load_vocab(file_path):
    word2idx = {}
    with open(file_path, "r") as f:
        for line in f:
            idx, word = line.strip().split()
            word2idx[word] = int(idx)
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
    word2idx = generate_vocab("../data/train")
    tag2idx = load_tags()
    print(word2idx)
    print(tag2idx)