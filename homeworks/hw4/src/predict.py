import torch
from torch.utils.data import DataLoader

from model import BiLSTMNER
from dataset import NERDataset, pad_collate
from vocab import load_vocab, load_tags

def predict(input_path, model_path="blstm1.pt", output_path="out/dev1.out"):
    word2idx = load_vocab(input_path)
    tag2idx = load_tags()
    idx2tag = {v: k for k, v in tag2idx.items()}

    dataset = NERDataset(input_path, word2idx, tag2idx)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=pad_collate)

    model = BiLSTMNER(len(word2idx), 100, 256, 128, len(tag2idx))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    predictions = []
    with torch.no_grad():
        for x_batch, _, x_lens in loader:
            outputs = model(x_batch, x_lens)
            pred_tags = outputs.argmax(dim=-1)
            for pred, length in zip(pred_tags, x_lens):
                predictions.append(pred[:length].cpu().numpy())

    with open(output_path, 'w') as f:
        for sent_idx, sentence in enumerate(dataset.sentences):
            for word_idx, word in enumerate(sentence):
                pred_tag = idx2tag[predictions[sent_idx][word_idx]]
                f.write(f"{word_idx+1} {word} {pred_tag}\n")
            f.write("\n")

    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    predict("data/dev")
