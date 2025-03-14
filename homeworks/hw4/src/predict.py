import torch
from torch.utils.data import DataLoader
import os
import argparse

from model import BiLSTMNER
from dataset import NERDataset, pad_collate
from vocab import load_vocab, load_tags
from glove import load_glove_embeddings

def predict(input_path, vocab_path, use_glove=False, has_tag=False, glove_path="./data/glove.6B.100d", model_path="blstm1.pt", output_path="out/dev1.out"):
    word2idx, idx2word = load_vocab(vocab_path)
    tag2idx = load_tags()
    idx2tag = {v: k for k, v in tag2idx.items()}

    print(f"Loading dataset from {input_path}...")
    dataset = NERDataset(input_path, word2idx, tag2idx, has_tag=has_tag)  
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=pad_collate)

    pretrained_embeddings = None
    if use_glove and os.path.exists(glove_path):
        print(f"Loading GloVe embeddings from {glove_path}...")
        pretrained_embeddings = load_glove_embeddings(glove_path, word2idx)

    model = BiLSTMNER(len(word2idx), 100, 256, 128, len(tag2idx), pretrained_embeddings=pretrained_embeddings)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print("Predicting...")
    predictions = []
    with torch.no_grad():
        for x_batch, _, x_lens in loader:  # Ignore labels (None in test mode)
            outputs = model(x_batch, x_lens)
            pred_tags = outputs.argmax(dim=-1)
            for pred, length in zip(pred_tags, x_lens):
                predictions.append(pred[:length].cpu().numpy())

    with open(output_path, 'w') as f:
        for sent_idx, sentence in enumerate(dataset.sentences):
            for word_idx, word in enumerate(sentence):
                word_text = idx2word.get(word, "<UNK>")
                pred_tag = idx2tag[predictions[sent_idx][word_idx]]
                f.write(f"{word_idx+1} {word_text} {pred_tag}\n")
            f.write("\n")

    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vocab_path", type=str, default="./out/vocab.txt", help="Path to vocabulary")
    parser.add_argument("-i", "--input_path", type=str, default="./data/dev", help="Path to data to predict")
    parser.add_argument("-m", "--model_path", type=str, default="./out/blstm1.pt", help="Path to model")
    parser.add_argument("--use_glove", action="store_true", help="Use GloVe embeddings")
    parser.add_argument("--has_tag", action="store_true", help="Input data has tags")
    parser.add_argument("--glove_path", type=str, default="./data/glove.6B.100d", help="Path to GloVe embeddings")
    parser.add_argument("-o", "--output_path", type=str, default="./out/dev1.out", help="Path to output data")
    args = parser.parse_args()
    
    predict(
        vocab_path=args.vocab_path,
        input_path=args.input_path,
        use_glove=args.use_glove,
        has_tag=args.has_tag,
        glove_path=args.glove_path,
        model_path=args.model_path,
        output_path=args.output_path
    )
