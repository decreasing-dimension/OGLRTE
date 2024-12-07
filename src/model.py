import shutil
import torch
import torch.nn.functional as F
import transformers
import math

from params import BERT_MODEL_PATH,  BERT_HIDDEN_LAYER_SIZE, MAX_LABEL_SIZE, LSTM_HIDDEN_SIZE, MAX_LEN

class Chomp1d(torch.nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(dropout)

        self.conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(dropout)

        self.net = torch.nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = torch.nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = torch.nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(torch.nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class BiLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_seq):
        packed_input = torch.nn.utils.rnn.pack_sequence(input_seq)
        output, hidden = self.lstm(packed_input)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        output = self.fc(output)
        return output

class GraphAttentionLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = torch.nn.Parameter(torch.empty(size=(in_features, out_features)))
        torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = torch.nn.Parameter(torch.empty(size=(2*out_features, 1)))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = torch.nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.mT
        return self.leakyrelu(e)

class GAT(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

class GraphConvolution(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class OGLRTE(torch.nn.Module):
    def __init__(self, node_num):
        super(OGLRTE, self).__init__()
        self.arg_alpha = 0.7
        self.node_num = node_num
        self.l1 = transformers.BertModel.from_pretrained(BERT_MODEL_PATH)
        self.GATLayer = GAT(nfeat=MAX_LEN + 1, nhid=LSTM_HIDDEN_SIZE, nclass=LSTM_HIDDEN_SIZE * 2, dropout=0.2, alpha=0.2, nheads=8)
        self.GCNLayer = GraphConvolution(MAX_LEN + 1, 2048, bias=False)
        self.GCNLayer2 = GraphConvolution(2048, 2048, bias=False)
        self.GCNLayer3 = GraphConvolution(2048, LSTM_HIDDEN_SIZE * 2, bias=False)
        self.relu = torch.nn.LeakyReLU(0.2)
        self.gat2cls = torch.nn.Linear(LSTM_HIDDEN_SIZE * 2, 1)
        self.dropout = torch.nn.Dropout(0.2)
        
        self.bilstm = torch.nn.LSTM(BERT_HIDDEN_LAYER_SIZE, LSTM_HIDDEN_SIZE, batch_first=True, bidirectional=True, dropout=0.2)
        self.w_omega = torch.nn.Parameter(torch.Tensor(LSTM_HIDDEN_SIZE * 2, LSTM_HIDDEN_SIZE * 2))
        self.u_omega = torch.nn.Parameter(torch.Tensor(LSTM_HIDDEN_SIZE * 2, 1))
        
        torch.nn.init.uniform_(self.w_omega, -0.1, 0.1)
        torch.nn.init.uniform_(self.u_omega, -0.1, 0.1)
        
        self.tcn = TemporalConvNet(num_inputs=BERT_HIDDEN_LAYER_SIZE, num_channels=[LSTM_HIDDEN_SIZE * 2, LSTM_HIDDEN_SIZE * 2, LSTM_HIDDEN_SIZE * 2, LSTM_HIDDEN_SIZE * 2, LSTM_HIDDEN_SIZE * 2], \
                                   kernel_size=2, dropout=0.2)
        self.w_omega2 = torch.nn.Parameter(torch.Tensor(LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE))
        self.u_omega2 = torch.nn.Parameter(torch.Tensor(LSTM_HIDDEN_SIZE, 1))
        
        torch.nn.init.uniform_(self.w_omega2, -0.1, 0.1)
        torch.nn.init.uniform_(self.u_omega2, -0.1, 0.1)
        self.nodecls = torch.nn.Linear(BERT_HIDDEN_LAYER_SIZE, node_num)
        
        self.fc = torch.nn.Linear(LSTM_HIDDEN_SIZE * 2, MAX_LABEL_SIZE)
        self.joint = torch.nn.Linear(MAX_LABEL_SIZE * 2, MAX_LABEL_SIZE)

        self.direct = torch.nn.Linear(BERT_HIDDEN_LAYER_SIZE, MAX_LABEL_SIZE)

    def forward(self, ids, mask, token_type_ids, ontology):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        last_hidden_state = output_1[0]
        batch = last_hidden_state.size()
        
        out, (h_n, c_n) = self.bilstm(last_hidden_state)
        out = self.dropout(out)
        
        output = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        out2 = self.tcn(last_hidden_state.permute(0, 2, 1))
        out2 = out2.permute(0, 2, 1)
        
        nodecls = last_hidden_state + torch.cat([out, out2], dim=2)
        nodecls = self.dropout(nodecls)
        nodecls = torch.nn.functional.relu(nodecls)
        nodecls = self.nodecls(nodecls)
        nodecls = self.dropout(nodecls)
        nodecls = nodecls.permute(0, 2, 1)
        nodecls = torch.nn.functional.sigmoid(nodecls)
        
        newbit, _ = torch.max(nodecls, dim=2)
        newbit = newbit.view(batch[0], -1, 1)
        
        nodecls = torch.cat((nodecls, newbit), dim=2)
        gat_output = self.GATLayer(nodecls, ontology["adj"])
        gat_output = self.gat2cls(gat_output).squeeze().view(-1, self.node_num)
        gat_output = gat_output[:, : MAX_LABEL_SIZE]
        
        merge_output = torch.cat((out, out2), dim = 1)
        merge_output = self.dropout(merge_output)
        
        u = torch.tanh(torch.matmul(merge_output, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)
        scored_x = merge_output * att_score
        feat = torch.sum(scored_x, dim=1)
        feat = self.fc(feat)
        
        return self.arg_alpha * feat + (1 - self.arg_alpha) * gat_output

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_path = best_model_path
        shutil.copyfile(f_path, best_path)
