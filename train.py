import torch
from rdkit import Chem
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import TransformerModel
from vocab import Vocabulary, MolData
import os

device = torch.device("cpu")

def autoregressive_train():
    # training config
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 10

    # load vocabulary
    vocab = Vocabulary("data/Voc.txt")
    print(f"Vocabulary size: {vocab.vocab_size}")

    # load dataset
    dataset = MolData("data/uspto.csv", vocab)  # CSV with product, reactant columns
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                             collate_fn=MolData.collate_fn)

    # init model
    model = TransformerModel(vocab).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")

    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        for step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            p_seqs, r_seqs = batch  # unpack tuple
            p_seqs = p_seqs.long().to(device)
            r_seqs = r_seqs.long().to(device)

            loss = model.compute_loss(p_seqs, r_seqs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if step % 15 == 0 and step != 0:
                tqdm.write("*" * 50)
                tqdm.write(f"Epoch {epoch}  step {step}  loss {loss.item():.4f}\n")

                # sample some sequences
                seqs_sample = model.sample(p_seqs[:5], max_len=200, temperature=1.0)
                valid = 0
                for i, seq in enumerate(seqs_sample):
                    smile = vocab.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        tqdm.write(smile)
                valid_percent = 100 * valid / len(seqs_sample)
                tqdm.write(f"{valid_percent:.1f}% valid SMILES")
                tqdm.write("*" * 50 + "\n")

                # save best model
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(model.state_dict(), "data/Retro.ckpt")
                    tqdm.write(f"New best model saved with loss: {best_loss:.4f}")

        # save checkpoint at end of epoch
        torch.save(model.state_dict(), f"data/Retro-{epoch}.ckpt")

if __name__ == '__main__':
    autoregressive_train()