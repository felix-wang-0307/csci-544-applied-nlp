import torch

def load_glove_embeddings(glove_path, word2idx, embedding_dim=100):
    """
    Load pretrained GloVe embeddings and align them with word2idx.
    """
    # Initialize embeddings with random values for missing words
    glove_embeddings = torch.empty(len(word2idx), embedding_dim).uniform_(-0.25, 0.25)
    glove_embeddings[word2idx["<PAD>"]] = torch.zeros(embedding_dim)  # Ensure padding is zeroed

    with open(glove_path, 'r', encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            word = parts[0]
            vector = torch.tensor([float(val) for val in parts[1:]], dtype=torch.float32)

            # Case-sensitive mapping: if lowercase word exists in word2idx, map it
            if word in word2idx:
                glove_embeddings[word2idx[word]] = vector
            elif word.lower() in word2idx:  # Handle case-insensitivity
                glove_embeddings[word2idx[word.lower()]] = vector

    return glove_embeddings

