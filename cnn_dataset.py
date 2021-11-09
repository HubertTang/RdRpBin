from torch.utils import data
import pandas as pd
import numpy as np


# dict to do onehot encoding for protein in pep-wise
PEP_BLOSUM62 = {'A': [0, 15], 'C': [1], 'D': [2, 3], 'E': [3, 8], 'F': [4, 18, 19], 'G': [5], 'H': [6, 19],
                'I': [7, 9, 10, 17], 'K': [8], 'L': [9, 10, 17], 'M': [10, 17], 'N': [11, 2, 6, 15], 'P': [12], 'Q': [13, 3, 8],
                'R': [14, 13, 8], 'S': [15, 16], 'T': [16], 'V': [17], 'W': [18, 19], 'Y': [19], 'X': [20],
                'O': [20], 'U': [20], 'B': [2, 3, 11, 2, 6, 15], 'Z': [3, 8, 13], 'J': [7, 10, 17, 9], "_": [], "*": []}


def encoding_blosum(seq, arr, seq_len=50):
    # arr = np.zeros((seq_length, 21))
    for i, c in enumerate(seq):
            
        for idx in PEP_BLOSUM62[c]:
            arr[0][i][idx] = 1


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
        encoding_blosum(self.seqs[index], seq_np, self.seq_len)
        return seq_np, self.labels[index]
