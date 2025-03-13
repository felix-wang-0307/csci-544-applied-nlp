import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import argparse

from model import BiLSTMNER
from dataset import NERDataset, pad_collate
from vocab import generate_vocab, load_tags
from glove import load_glove_embeddings


def train_model(
    train_path,
    dev_path,
    vocab_path="./out/vocab.txt",
    use_glove=False,
    glove_path="./data/glove.6B.100d.gz",
    save_model_path="./out/blstm1.pt",
    epochs=10,
    batch_size=32,
    learning_rate=0.05,
):
    # Load vocabulary and tags
    word2idx = generate_vocab(train_path, vocab_path)
    tag2idx = load_tags()

    # Load datasets
    train_dataset = NERDataset(train_path, word2idx, tag2idx)
    dev_dataset = NERDataset(dev_path, word2idx, tag2idx)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate
    )

    pretrained_embeddings = None
    if use_glove:
        pretrained_embeddings = load_glove_embeddings(glove_path, word2idx)

    # Model
    model = BiLSTMNER(
        vocab_size=len(word2idx),
        embedding_dim=100,
        hidden_dim=256,
        linear_dim=128,
        tagset_size=len(tag2idx),
        pretrained_embeddings=pretrained_embeddings
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch, x_lens in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch, x_lens)
            loss = criterion(outputs.view(-1, len(tag2idx)), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(train_loader):.4f}")

    if not os.path.exists(os.path.dirname(save_model_path)):
        os.makedirs(os.path.dirname(save_model_path))

    torch.save(model.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--train_path",
        type=str,
        default="./data/train",
        help="Path to training data",
    )
    parser.add_argument(
        "-d", "--dev_path", type=str, default="./data/dev", help="Path to dev data"
    )
    parser.add_argument(
        "-v", "--vocab_path", type=str, default="./out/vocab.txt", help="Path to saved vocabulary"
    )
    parser.add_argument(
        "-g", "--use_glove", action="store_true", help="Use GloVe embeddings"
    )
    parser.add_argument(
        "--glove_path",
        type=str,
        default="./data/glove.6B.100d.gz",
        help="Path to GloVe embeddings"
    )

    parser.add_argument(
        "--save_model_path",
        type=str,
        default="./out/blstm1.pt",
        help="Path to save model",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.05, help="Learning rate"
    )
    args = parser.parse_args()

    train_model(
        args.train_path,
        args.dev_path,
        args.vocab_path,
        args.use_glove,
        args.glove_path,
        args.save_model_path,
        args.epochs,
        args.batch_size,
        args.learning_rate,
    )
