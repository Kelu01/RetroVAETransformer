import torch
from torch import nn
from torch.nn import functional as F

class VAETransformer(nn.Module):
    def __init__(self, vocab_size, d_model, latent_dim, n_layers, n_heads, dropout, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)  # token embedding
        self.position_embedding = nn.Embedding(max_len, d_model)  # positional embedding

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True  # batch first for easier indexing
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.to_mean = nn.Linear(d_model, latent_dim)  # latent mean
        self.to_logvar = nn.Linear(d_model, latent_dim)  # latent logvar

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.latent_to_hidden = nn.Linear(latent_dim, d_model)  # latent → decoder hidden
        self.output_layer = nn.Linear(d_model, vocab_size)  # logits

    def encode(self, src, pad_token_id=0):
        """
        Encode input sequence into latent mean and logvar.
        src: (batch, seq) token ids
        """
        src_mask = (src == pad_token_id)
        seq_len = src.size(1)
        positions = torch.arange(0, seq_len, device=src.device).unsqueeze(0)
        x = self.token_embedding(src) + self.position_embedding(positions)
        encoded = self.encoder(x, src_key_padding_mask=src_mask)
        mask = ~src_mask
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        masked_encoded = encoded * mask.unsqueeze(-1)
        mean_vec = masked_encoded.sum(dim=1) / lengths
        mean = self.to_mean(mean_vec)
        logvar = self.to_logvar(mean_vec)
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """
        Sample latent vector using reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def _causal_mask(self, seq_len, device):
        """
        Create upper-triangular mask to prevent attending to future tokens.
        """
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
        return mask

    def decode(self, tgt, z, pad_token_id=0):
        """
        Decode latent vector z into output token probabilities.
        """
        tgt_mask = (tgt == pad_token_id)
        seq_len = tgt.size(1)
        positions = torch.arange(0, seq_len, device=tgt.device).unsqueeze(0)
        x = self.token_embedding(tgt) + self.position_embedding(positions)
        memory = self.latent_to_hidden(z).unsqueeze(1).repeat(1, seq_len, 1)
        causal_mask = self._causal_mask(seq_len, tgt.device)
        decoded = self.decoder(
            x,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_mask
        )
        return self.output_layer(decoded)

    def forward(self, src, tgt, pad_token_id=0):
        """
        Forward pass: encode src, sample latent z, decode tgt.
        """
        mean, logvar = self.encode(src, pad_token_id)
        z = self.reparameterize(mean, logvar)
        output = self.decode(tgt, z, pad_token_id)
        return output, mean, logvar

class TransformerModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.pad_token_id = vocab.vocab["[PAD]"]
        self.start_token_id = vocab.vocab["[GO]"]
        self.eos_token_id = vocab.vocab["[EOS]"]

        self.model = VAETransformer(
            vocab_size=vocab.vocab_size,
            d_model=512,
            latent_dim=256,
            n_layers=8,
            n_heads=16,
            dropout=0.1,
            max_len=1300
        )

    def compute_loss(self, input_ids, target_ids, kl_weight=0.1):
        """
        Compute reconstruction + KL loss for VAE.
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        go_col = torch.full((batch_size, 1), self.start_token_id, device=device, dtype=torch.long)
        decoder_input = torch.cat([go_col, target_ids[:, :-1].long()], dim=1)

        output, mean, logvar = self.model(input_ids, decoder_input, pad_token_id=self.pad_token_id)
        logvar = torch.clamp(logvar, min=-10, max=10)

        reconstruction_loss = F.cross_entropy(
            output.reshape(-1, output.size(-1)),
            target_ids.reshape(-1).long(),
            ignore_index=self.pad_token_id
        )

        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        loss = reconstruction_loss + kl_loss
        return loss

    @torch.no_grad()
    def sample(self, input_ids, max_len=1300, temperature=1.0):
        """
        Autoregressive sampling from the VAE-Transformer.
        input_ids: (batch, seq) product tokens
        """
        mean, logvar = self.model.encode(input_ids, pad_token_id=self.pad_token_id)
        z = self.model.reparameterize(mean, logvar)

        batch_size = input_ids.size(0)
        device = input_ids.device

        seq = torch.full((batch_size, 1), self.start_token_id, device=device, dtype=torch.long)
        for _ in range(max_len):
            logits = self.model.decode(seq, z, pad_token_id=self.pad_token_id)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)

            next_token = torch.argmax(probs, dim=-1, keepdim=True)

            seq = torch.cat([seq, next_token], dim=1)
            if (next_token == self.eos_token_id).all():
                break
        return seq