import torch
import torch.nn as nn
import torch.nn.functional as F

import random


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        query = query.unsqueeze(1)  # [batch, 1, hid_dim]

        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2)#.unsqueeze(1)

        weights = F.softmax(scores, dim=1).unsqueeze(1)  # Make weights 3D: [batch_size, 1, src_sent_len]
        context = torch.bmm(weights, keys)

        return context, weights


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src):

        embedded = self.embedding(src)

        embedded = self.dropout(embedded)

        output, (hidden, cell) = self.rnn(embedded)

        return output, hidden, cell
    

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )

        # The input size for the RNN should be emb_dim + hid_dim to accommodate the concatenated context vector
        self.rnn = nn.LSTM(
            input_size=emb_dim + hid_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )

        self.out = nn.Linear(
            in_features=hid_dim,
            out_features=output_dim
        )

        self.dropout = nn.Dropout(p=dropout)
        self.attention = BahdanauAttention(hid_dim)


    def forward(self, input, hidden, cell, encoder_outputs):

        # input = [batch size]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        # encoder_outputs = [src sent len, batch size, hid dim]

        input = input.unsqueeze(0) # [1, batch size]

        # embedded = [1, batch size, emb dim]
        embedded = self.dropout(self.embedding(input))

        # query = [batch size, hid dim]
        query = hidden[-1].squeeze(0) # Take the last layer's hidden state and remove the layer dimension

        # keys = [batch size, src sent len, hid dim]
        keys = encoder_outputs.permute(1, 0, 2)

        # context = [batch size, 1, hid dim], attn_weights = [batch size, src sent len]
        context, attn_weights = self.attention(query, keys)

        # context = [1, batch size, hid dim]
        context = context.permute(1, 0, 2)

        # rnn_input = [1, batch size, emb dim + hid dim]
        rnn_input = torch.cat((embedded, context), dim=2)

        # Pass rnn_input (with attention context) to the RNN
        # output = [1, batch size, hid dim]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # prediction = [batch size, output dim]
        prediction = self.out(output.squeeze(0))


        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):

        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        # hidden, cell = self.encoder(src)
        encoder_outputs, hidden, cell = self.encoder(src)

        #first input to the decoder is the <sos> tokens
        input = trg[0,:]

        for t in range(1, max_len):

            # output, hidden, cell = self.decoder(input, hidden, cell)
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)

        return outputs