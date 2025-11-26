import numpy as np
import re 
import torch
from torch.utils.data import Dataset
import pandas as pd


class Vocabulary(object):
    def __init__(self, file, max_length=1300):
        self.special_tokens = ["[PAD]", "[EOS]", "[GO]"]
        self.additional_chars = self.init_from_file(file)
        self.chars = self.special_tokens + self.additional_chars
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length

    def init_from_file(self, file):
        with open(file, 'r') as f:
            chars = f.read().split()
        return chars

    def tokenize(self, smiles):
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = self.replace_halogen(smiles)
        parts = re.split(regex, smiles)
        tokenized = []
        for p in parts:
            if p.startswith('['):
                tokenized.append(p)
            else:
                for c in p:
                    tokenized.append(c)
        tokenized.append("[EOS]")
        return tokenized

    def encode(self, token_list):
        t = torch.zeros(len(token_list), dtype=torch.long)
        for i, tok in enumerate(token_list):
            t[i] = self.vocab[tok]
        return t

    def decode(self, arr):
        arr = [int(i) for i in arr]
        out = []
        for i in arr:
            if i == self.vocab["[EOS]"]:
                break
            if i == self.vocab["[GO]"]:
                continue
            out.append(self.reversed_vocab[i])
        smiles = "".join(out)
        smiles = self.restore_halogen(smiles)
        return smiles

    def replace_halogen(self, smiles):
        return smiles.replace("Cl", "L").replace("Br", "R")

    def restore_halogen(self, smiles):
        # Replace L → Cl when NOT inside brackets
        smiles = re.sub(r'(?<!\[)L(?!\])', 'Cl', smiles)
        # Replace R → Br when NOT inside brackets
        smiles = re.sub(r'(?<!\[)R(?!\])', 'Br', smiles)
        return smiles


class MolData(Dataset):
    def __init__(self, csv_file, voc):
        self.voc = voc
        df = pd.read_csv(csv_file)
        self.products = df["product"].astype(str).tolist()
        self.reactants = df["reactant"].astype(str).tolist()

    def __getitem__(self, index):
        p = self.products[index]
        r = self.reactants[index]
        p_tokens = self.voc.tokenize(p)
        r_tokens = self.voc.tokenize(r)
        p_enc = self.voc.encode(p_tokens)
        r_enc = self.voc.encode(r_tokens)
        return p_enc.clone().detach(), r_enc.clone().detach()

    def __len__(self):
        return len(self.products)

    @classmethod
    def collate_fn(cls, arr):
        p_seqs = [x[0] for x in arr]
        r_seqs = [x[1] for x in arr]
        p_max = max([seq.size(0) for seq in p_seqs])
        r_max = max([seq.size(0) for seq in r_seqs])
        p_batch = torch.zeros(len(arr), p_max)
        r_batch = torch.zeros(len(arr), r_max)
        for i in range(len(arr)):
            p_batch[i, :p_seqs[i].size(0)] = p_seqs[i]
            r_batch[i, :r_seqs[i].size(0)] = r_seqs[i]
        return p_batch, r_batch
    
