from torch.utils import data
import pandas as pd
import numpy as np
from Bio.Seq import translate
from Bio import SeqIO
import pickle
import random


# dict to do onehot encoding for protein in pep-wise
PEPSET = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6,
          'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13,
          'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20,
          'O': 20, 'U': 20, 'B': (2, 11), 'Z': (3, 13), 'J': (7, 9)}

PEP_BLOSUM62 = {'A': [0, 15], 'C': [1], 'D': [2, 3], 'E': [3, 8], 'F': [4, 18, 19], 'G': [5], 'H': [6, 19],
                'I': [7, 9, 10, 17], 'K': [8], 'L': [9, 10, 17], 'M': [10, 17], 'N': [11, 2, 6, 15], 'P': [12], 'Q': [13, 3, 8],
                'R': [14, 13, 8], 'S': [15, 16], 'T': [16], 'V': [17], 'W': [18, 19], 'Y': [19], 'X': [20],
                'O': [20], 'U': [20], 'B': [2, 3, 11, 2, 6, 15], 'Z': [3, 8, 13], 'J': [7, 10, 17, 9], "_": [], "*": []}

DNASET = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'K': (2, 3), 'M': (0, 1), 
          'R': (0, 2), 'Y': (1, 3), 'S': (1, 2), 'W': (0, 3),
          'B': (1, 2, 3), 'V': (0, 1, 2), 'H': (0, 1, 3), 'D': (0, 2, 3), 
          'X': (0, 1, 2, 3), 'N': (0, 1, 2, 3)}


def encoding_pep_onehot(seq, arr, seq_len=50):
    # arr = np.zeros((seq_length, 21))
    for i, c in enumerate(seq):
        if i < seq_len:
            if c == "_" or c == "*":
                # let them zero
                continue
            
            elif isinstance(PEPSET[c], int):
                idx = PEPSET[c]
                arr[0][i][idx] = 1
            
            else:
                idx1 = PEPSET[c][0]
                idx2 = PEPSET[c][1]
                arr[0][i][idx1] = 0.5
                arr[0][i][idx2] = 0.5


def encoding_blosum(seq, arr, seq_len=50):
    # arr = np.zeros((seq_length, 21))
    for i, c in enumerate(seq):
            
        for idx in PEP_BLOSUM62[c]:
            arr[0][i][idx] = 1


def encoding_dna_onehot(seq, arr, seq_len=150):
    # arr = np.zeros((seq_length, 150))
    for i, c in enumerate(seq):
        if i < seq_len:
            if c == "_" or c == "*":
                # let them zero
                continue
            
            elif isinstance(DNASET[c], int):
                idx = DNASET[c]
                arr[0][i][idx] = 1
            
            else:
                nums = len(DNASET[c])
                for idx in DNASET[c]:
                    arr[0][i][idx] = 1 / nums


class PepOnehot(data.Dataset):
    def __init__(self, file, type='train', seq_len = 50):
        self.type = type
        self.file = file
        self.seq_len = seq_len
        # column 0 is label, column 1 is seq
        df = pd.read_csv(self.file, sep=',', header = None)
        self.labels = df[0]
        self.seqs = df[1]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq_np = np.zeros((1, self.seq_len, 21), dtype=np.float32)
        encoding_pep_onehot(self.seqs[index], seq_np, self.seq_len)
        return seq_np, self.labels[index]


class PepOnehot_blosum(data.Dataset):
    def __init__(self, file, type='train', seq_len = 50):
        self.type = type
        self.file = file
        self.seq_len = seq_len
        # column 0 is label, column 1 is seq
        df = pd.read_csv(self.file, sep=',', header = None)
        self.labels = df[0]
        self.seqs = df[1]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq_np = np.zeros((1, self.seq_len, 21), dtype=np.float32)
        encoding_blosum(self.seqs[index], seq_np, self.seq_len)
        return seq_np, self.labels[index]


class PepOnehot_simese(data.Dataset):
    def __init__(self, file, type='train', seq_len = 50):
        self.type = type
        self.file = file
        self.seq_len = seq_len
        # column 0 is label, column 1 is seq
        df = pd.read_csv(self.file, sep=',', header = None)
        self.labels = df[0]
        self.seqs1 = df[1]
        self.seqs2 = df[2]

    def __len__(self):
        return len(self.seqs1)

    def __getitem__(self, index):
        seq_np1 = np.zeros((1, self.seq_len, 21), dtype=np.float32)
        seq_np2 = np.zeros((1, self.seq_len, 21), dtype=np.float32)
        encoding_pep_onehot(self.seqs1[index], seq_np1, self.seq_len)
        encoding_pep_onehot(self.seqs2[index], seq_np2, self.seq_len)
        return seq_np1, seq_np2, self.labels[index]


class DnaOnehot(data.Dataset):
    def __init__(self, file, type='train', seq_len = 150):
        self.type = type
        self.file = file
        self.seq_len = seq_len
        # column 0 is label, column 1 is seq
        df = pd.read_csv(self.file, sep=',', header = None)
        self.labels = df[0]
        self.seqs = df[1]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq_np = np.zeros((1, self.seq_len, 4), dtype=np.float32)
        encoding_dna_onehot(self.seqs[index], seq_np, self.seq_len)
        return seq_np, self.labels[index]

