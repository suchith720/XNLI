import torch
from torch import nn
import torch.nn.functional as F

class MLP(torch.nn.Module):

    def __init__(self, layers, device, p=0.5):
        super(MLP, self).__init__()

        self.device = device
        self.layers = layers

        self.lin = {}
        for i in range(1, len(layers)):
            self.lin[f'lin_{i-1}'] = torch.nn.Linear(layers[i-1],
                                                         layers[i],
                                                         device=device)
        self.dropout = torch.nn.Dropout(p=p)
        self.act = torch.nn.ReLU()

        self.params = nn.ModuleDict({
            'params': nn.ModuleList([self.lin[f'lin_{i}'] for i in range(len(self.lin))]),
                                     })

    def forward(self, x):
        for i in range(len(self.lin)):
            x = self.lin[f'lin_{i}'](x)
            if i != len(self.lin)-1:
                x = self.act(x)
                x = self.dropout(x)

        return x


class XLMRXLNIModel(nn.Module):

    def __init__(self, model, device, lstm_params, attention_params,
                 dropout_params, layers=[], p=0.5):

        super(XLMRXLNIModel, self).__init__()

        self.device = device

        self.xlmr = model.to(device)
        self.drop1 = nn.Dropout(p=dropout_params['xlmr_drop'])

        self.lstm = nn.LSTM(**lstm_params)
        self.drop2 = nn.Dropout(p=dropout_params['lstm_drop'])
        self.attention = nn.MultiheadAttention(**attention_params)
        self.drop3 = nn.Dropout(p=dropout_params['attn_drop'])

        layers = [model.config.to_dict()['hidden_size']*2] + layers
        self.fc = MLP(layers, device, dropout_params['mlp_drop'])

        self.params = nn.ModuleDict({
            'group_1': nn.ModuleList([self.xlmr]),
            'group_2': nn.ModuleList([self.lstm, self.attention]),
            'group_3': self.fc.params['params']
                                     })

    def freeze_layer(self, group='group_1'):
        for module in self.params[group]:
            for param in module.parameters():
                module.param = False

    def unfreeze_layer(self, group='group_1'):
        for module in self.params[group]:
            for param in module.parameters():
                module.param = True

    def forward(self, x):
        o = self.xlmr(input_ids=x['input_ids'],
                      attention_mask=x['attention_mask'])
        o = o.last_hidden_state
        o = self.drop1(o)

        bos = o[:, 0:1]
        lstm_in = o[:, 1:]

        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = self.drop2(lstm_out)
        attn_out, _ = self.attention(bos, lstm_out, lstm_out)
        attn_out = self.drop3(attn_out)

        bos = bos.squeeze()
        attn_out = attn_out.squeeze()

        o = torch.concat([bos, attn_out], dim=1)
        o = self.fc(o)

        return F.log_softmax(o, dim=1)


