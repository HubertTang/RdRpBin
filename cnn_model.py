import torch
import torch.nn as nn
import torch.nn.functional as F


class RdRpBinCNN(nn.Module):
    # use CNN to classify RdRp sequences
    def __init__(self, num_class=5, num_token=21, seq_len=50, kernel_nums=256,
                 kernel_sizes=[2, 4, 6, 8, 10, 12], dropout=0.5, num_fc=512, out_logit=True):
        super(RdRpBinCNN, self).__init__()

        self.num_token = num_token
        self.seq_len = seq_len
        self.num_class = num_class
        self.channle_in = 1
        self.kernel_nums = kernel_nums
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout
        self.out_logit = out_logit

        self.convs1 = nn.ModuleList(
            [nn.Conv2d(self.channle_in, self.kernel_nums, kernel_size= (kernel_size, self.num_token)) 
             for i, kernel_size in enumerate(self.kernel_sizes)])

        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(self.kernel_nums * len(self.kernel_sizes), num_fc)
        self.fc2 = nn.Linear(num_fc, self.num_class)

    def forward(self, x):
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)
        
        if self.out_logit:
            x = self.fc1(x)
            logit = self.fc2(x)
            return logit
        else:
            fc_1 = self.fc1(x)
            logit = self.fc2(fc_1)
            return fc_1
