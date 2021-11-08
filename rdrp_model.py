import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DeepRdRp(nn.Module):
    # use CNN to classify RdRp sequences
    def __init__(self, num_class=5, num_token=21, seq_len=50, kernel_nums=256,
                 kernel_sizes=[2, 4, 6, 8, 10, 12], dropout=0.5, num_fc=512, out_logit=True):
        super(DeepRdRp, self).__init__()

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


class DeepRdRp_noPooling(nn.Module):
    # use CNN to classify RdRp sequences
    def __init__(self, num_class=5, num_token=21, seq_len=66, kernel_nums=256,
                 kernel_sizes=[2, 4, 6, 8, 10, 12], dropout=0.5, num_fc=512):
        super(DeepRdRp_noPooling, self).__init__()

        self.num_token = num_token
        self.seq_len = seq_len
        self.num_class = num_class
        self.channle_in = 1
        self.kernel_nums = kernel_nums
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout

        self.convs1 = nn.ModuleList(
            [nn.Conv2d(self.channle_in, self.kernel_nums, kernel_size= (kernel_size, self.num_token)) 
             for i, kernel_size in enumerate(self.kernel_sizes)])

        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(self.kernel_nums * len(self.kernel_sizes) * 42, num_fc)
        self.fc2 = nn.Linear(num_fc, self.num_class)

    def forward(self, x):
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        # x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        # x = torch.cat(x, 1)
        # for i in x:
        #     print(i.size())
        x = [i.view(i.size(0), -1) for i in x]
        # for i in x:
        #     print(i.size())
        # exit(0)
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        logit = self.fc2(x)
        return logit


class DeepRdRp_BiLSTM(nn.Module):
    # use LSTM to classify RdRp sequences
    def __init__(self, num_class=5, num_token=21, seq_len=50, kernel_nums=256,
                 dropout=0.5, num_fc=512):
        super(DeepRdRp_BiLSTM, self).__init__()

        self.bilstm = nn.LSTM(num_token, num_fc // 2, num_layers=1, 
                              dropout=dropout, bidirectional=True, bias=False)

        self.fc1 = nn.Linear(num_fc, num_fc // 2)
        self.fc2 = nn.Linear(num_fc // 2, num_class)

    def forward(self, x):
        
        bilstm_out, _ = self.bilstm(x)

        
class DeepRdRp_penult(nn.Module):
    # use CNN to classify RdRp sequences
    def __init__(self, num_class=5, num_token=21, seq_len=50, kernel_nums=256,
                 kernel_sizes=[2, 4, 6, 8, 10, 12], dropout=0.5, num_fc=512):
        super(DeepRdRp_penult, self).__init__()

        self.num_token = num_token
        self.seq_len = seq_len
        self.num_class = num_class
        self.channle_in = 1
        self.kernel_nums = kernel_nums
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout

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
        penult = self.fc1(x)
        logit = self.fc2(penult)
        return penult, logit


class DeepRdRp_double_dropout(nn.Module):
    # use CNN to classify RdRp sequences
    def __init__(self, num_class=5, num_token=21, seq_len=50, kernel_nums=256,
                 kernel_sizes=[2, 4, 6, 8, 10, 12], dropout=0.5, num_fc=512):
        super(DeepRdRp_double_dropout, self).__init__()

        self.num_token = num_token
        self.seq_len = seq_len
        self.num_class = num_class
        self.channle_in = 1
        self.kernel_nums = kernel_nums
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout

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
        x = self.fc1(x)
        x = self.dropout(x)
        logit = self.fc2(x)
        return logit


class DeepRdRp_noise_nn(nn.Module):
    # use CNN to classify RdRp sequences
    def __init__(self, num_class=5, num_token=21, seq_len=50, kernel_nums=256,
                 kernel_sizes=[2, 4, 6, 8, 10, 12], dropout=0.5, num_fc=512):
        super(DeepRdRp_noise_nn, self).__init__()

        self.num_token = num_token
        self.seq_len = seq_len
        self.num_class = num_class
        self.channle_in = 1
        self.kernel_nums = kernel_nums
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout

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
        x = self.fc1(x)
        if self.training:
            shape = x.size()
            noise = torch.cuda.FloatTensor(shape) if torch.cuda.is_available() else torch.FloatTensor(shape)
            torch.rand(shape, out=noise)
            x  = x + x * (1 - 0.5 * noise)/ 100
            # torch.randn(shape, out=noise)
            # x += noise*0.01
        logit = self.fc2(x)
        return logit


class DeepRdRp_attention(nn.Module):
    # use CNN to classify RdRp sequences
    """https://github.com/avinashsai/Attention-based-CNN-for-sentence-classification
    """
    def __init__(self, num_class=5, num_token=21, seq_len=50, kernel_nums=256,
                 kernel_sizes=[2, 4, 6, 8, 10, 12], dropout=0.5, num_fc=512, 
                 atthidden=128, lamda=0.0):
        super(DeepRdRp_attention, self).__init__()

        self.num_token = num_token
        self.seq_len = seq_len
        self.num_class = num_class
        self.channle_in = 1
        self.kernel_nums = kernel_nums
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout
        self.atthidden = atthidden
        self.lamda = lamda

        self.attlinear1 = nn.Linear(self.num_token*2, self.atthidden)
        self.attlinear2 = nn.Linear(self.atthidden,1)

        self.convs1 = nn.ModuleList(
            [nn.Conv2d(self.channle_in, self.kernel_nums, kernel_size= (kernel_size, self.num_token*2)) 
             for i, kernel_size in enumerate(self.kernel_sizes)])

        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(self.kernel_nums * len(self.kernel_sizes), num_fc)
        self.fc2 = nn.Linear(num_fc, self.num_class)

        self.tanh = nn.Tanh()

    def attention_word(self, orig_input, curindex):
        input_1 = orig_input[:, curindex, :].unsqueeze(1).repeat(1, self.seq_len - 1, 1)
        input_2 = torch.cat([orig_input[:, :curindex, :], orig_input[:, curindex+1:, :]], dim=1)
        input = torch.cat([input_1, input_2], dim=2)
        input = torch.reshape(input, (-1, self.num_token*2))
        attout = self.tanh(self.attlinear1(input)).view(-1, self.seq_len - 1, self.atthidden)

        if curindex == 0:
            # score = torch.stack([math.pow((1-self.lamda), abs(curindex - index) - 1)*attout[:, index-1] for index in range(curindex+1, self.seq_len)], 1)
            score = torch.stack([attout[:, index-1] for index in range(curindex+1, self.seq_len)], 1)
        elif curindex == self.seq_len - 1:
            # score = torch.stack([math.pow((1-self.lamda), abs(curindex - index) - 1)*attout[:, index] for index in range(0, curindex)], 1)
            score = torch.stack([attout[:, index] for index in range(0, curindex)], 1)
        else:
            # score1 = torch.stack([math.pow((1-self.lamda), abs(curindex - index) - 1)*attout[:, index] for index in range(0, curindex)], 1)
            # score2 = torch.stack([math.pow((1-self.lamda), abs(curindex - index) - 1)*attout[:, index] for index in range(curindex+1, self.seq_len)], 1)
            score1 = torch.stack([attout[:, index] for index in range(0, curindex)], 1)
            score2 = torch.stack([attout[:, index-1] for index in range(curindex+1, self.seq_len)], 1)
            score = torch.cat([score1, score2], dim=1)

        attout = self.attlinear2(score).squeeze(2)
        alpha = F.softmax(attout, 1)
        out = alpha.unsqueeze(2) * input_2
        out = torch.sum(out, 1)
        return out

    def attention(self, input_x):
        x = input_x[:, 0, :, :]
        att_x = torch.stack([self.attention_word(x, i) for i in range(self.seq_len)], dim=1)
        return att_x

    def forward(self, x):
        att_x = self.attention(x)
        x = torch.cat((x.squeeze(1), att_x), dim=2).unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        logit = self.fc2(x)
        return logit


class DeepRdRp_nofinal(nn.Module):
    # use CNN to classify RdRp sequences
    def __init__(self, num_class=5, num_token=21, seq_len=50, kernel_nums=256,
                 kernel_sizes=[2, 4, 6, 8, 10, 12], dropout=0.5, num_fc=512):
        super(DeepRdRp_nofinal, self).__init__()

        self.num_token = num_token
        self.seq_len = seq_len
        self.num_class = num_class
        self.channle_in = 1
        self.kernel_nums = kernel_nums
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout

        self.convs1 = nn.ModuleList(
            [nn.Conv2d(self.channle_in, self.kernel_nums, kernel_size= (kernel_size, self.num_token)) 
             for i, kernel_size in enumerate(self.kernel_sizes)])

        self.dropout = nn.Dropout(self.dropout_rate)
        # self.fc1 = nn.Linear(self.kernel_nums * len(self.kernel_sizes), num_fc)
        self.fc1 = nn.Sequential(nn.Linear(self.kernel_nums * len(self.kernel_sizes), num_fc), nn.Sigmoid())
        # self.fc2 = nn.Linear(num_fc, self.num_class)

    def forward(self, x):
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x


class DeepRdRp_siamese_bce(nn.Module):
    # use CNN to classify RdRp sequences
    def __init__(self, num_class=5, num_token=21, seq_len=50, kernel_nums=256,
                 kernel_sizes=[2, 4, 6, 8, 10, 12], dropout=0.5, num_fc=512):
        super(DeepRdRp_siamese_bce, self).__init__()

        self.num_token = num_token
        self.seq_len = seq_len
        self.num_class = num_class
        self.channle_in = 1
        self.kernel_nums = kernel_nums
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout

        self.convs1 = nn.ModuleList(
            [nn.Conv2d(self.channle_in, self.kernel_nums, kernel_size= (kernel_size, self.num_token)) 
             for i, kernel_size in enumerate(self.kernel_sizes)])

        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(self.kernel_nums * len(self.kernel_sizes), num_fc)
        self.fc2 = nn.Linear(num_fc * 2, self.num_class)

    def forward_1(self, x):
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        # logit = self.fc2(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_1(x1)
        out2 = self.forward_1(x2)
        x = torch.cat([out1, out2], dim=1)
        print(x.size())
        logit = self.fc2(x)
        return logit


class DeepRdRp_embed_LSTM_att(nn.Module):
    # use LSTM to classify RdRp sequences
    def __init__(self, num_class=5, num_token=21, seq_len=50, batch_size=1024,
                 dropout=0.5, num_fc=512, vocab_size=10000, use_cuda=True):
        super(DeepRdRp_embed_LSTM_att, self).__init__()

        self.num_token = num_token
        self.seq_len = seq_len
        self.num_class = num_class
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.hidden_size = num_fc
        self.layer_size = 2
        self.attention_size = num_fc
        self.batch_size = batch_size

        self.lookup_table = nn.Embedding(self.vocab_size, self.num_token, padding_idx=0)
        # self.lookup_table.weight.data.uniform_(-1., 1.)

        self.bilstm = nn.LSTM(num_token, self.hidden_size, num_layers=1, 
                              dropout=self.dropout, bidirectional=True)

        if self.use_cuda:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size).cuda())
            self.u_omega = Variable(torch.zeros(self.attention_size).cuda())
        else:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size))
            self.u_omega = Variable(torch.zeros(self.attention_size))
        
        self.fc1 = nn.Linear(self.hidden_size * self.layer_size, num_fc*10)
        self.fc2 = nn.Linear(num_fc*10, num_fc*10)
        self.fc3 = nn.Linear(num_fc*10, num_class)

    def attention_net(self, lstm_output):
        #print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size*2])
        #print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        #print(attn_tanh.size()) = (squence_length * batch_size, attention_size)
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        #print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.seq_len])
        #print(exps.size()) = (batch_size, squence_length)
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        #print(alphas.size()) = (batch_size, squence_length)
        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.seq_len, 1])
        #print(alphas_reshape.size()) = (batch_size, squence_length, 1)
        state = lstm_output.permute(1, 0, 2)
        #print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)
        attn_output = torch.sum(state * alphas_reshape, 1)
        #print(attn_output.size()) = (batch_size, hidden_size*layer_size)

        return attn_output
    
    def forward(self, x):

        input = self.lookup_table(x)
        input = input.permute(1, 0, 2)

        if self.use_cuda:
            h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
        else:
            h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))

        lstm_output, (final_hidden_state, final_cell_state) = self.bilstm(input, (h_0, c_0))
        attn_output = self.attention_net(lstm_output)

        fc1_out = F.relu(self.fc1(attn_output))
        fc2_out = F.relu(self.fc2(fc1_out))
        logits = self.fc3(fc2_out)

        return logits


class ProtCNN():
    pass
