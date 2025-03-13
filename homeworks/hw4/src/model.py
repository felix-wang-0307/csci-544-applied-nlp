import torch
import torch.nn as nn

class BiLSTMNER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, linear_dim, tagset_size, pretrained_embeddings=None):
        super(BiLSTMNER, self).__init__()

        # Embedding layer: either learnable or initialized with GloVe embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)

        self.bilstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=1, 
            bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_dim * 2, linear_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(linear_dim, tagset_size)

    def forward(self, x, x_lens):
        embedded = self.embedding(x)
        packed_emb = nn.utils.rnn.pack_padded_sequence(embedded, x_lens, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm(packed_emb)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        linear_out = self.elu(self.linear(lstm_out))
        logits = self.classifier(linear_out)
        return logits
