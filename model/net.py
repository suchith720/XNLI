import torch
from torch import nn
from torchcrf import CRF
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class NERNetPOS(nn.Module):

    def __init__(self, token_embed_params, token_p,
                 char_embed_params, char_p,
                 pos_embed_params, pos_p,
                 isCap_embed_params, isCap_p,
                 isNum_embed_params, isNum_p,
                 hidden_dim, num_class, scale_grad_by_freq=False,
                 num_layers=1, dropout=0, proj_size=0, device=None):

        super(NERNetPOS, self).__init__()

        self.token_embed = nn.Embedding(**token_embed_params,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)
        self.token_dropout = nn.Dropout(p=token_p)


        self.pos_embed = nn.Embedding(**pos_embed_params,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)
        self.pos_dropout = nn.Dropout(p=pos_p)

        self.char_embed = nn.Embedding(**char_embed_params,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)
        self.char_dropout = nn.Dropout(p=char_p)

        """
        Binary features
        """
        self.isCap_embed = nn.Embedding(**isCap_embed_params,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)
        self.isCap_dropout = nn.Dropout(p=isCap_p)

        self.isNum_embed = nn.Embedding(**isNum_embed_params,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)
        self.isNum_dropout = nn.Dropout(p=isNum_p)

        embed_dim = token_embed_params['embedding_dim'] + \
                    pos_embed_params['embedding_dim'] + \
                    char_embed_params['embedding_dim'] + \
                    isCap_embed_params['embedding_dim'] + \
                    isNum_embed_params['embedding_dim']

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout, proj_size=proj_size, device=device)
        """
        Sets forget gate bias
        """
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)


        self.fc = nn.Linear(2*hidden_dim, num_class,
                            device=device) if proj_size == 0 else nn.Linear(2*proj_size, num_class, device=device)

        self.params = nn.ModuleDict({
            'group_1': nn.ModuleList([self.token_embed, self.pos_embed,
                                      self.char_embed, self.isCap_embed, self.isNum_embed]),
            'group_2': nn.ModuleList([self.lstm, self.fc])})


    def forward(self, x, x_len):
        xt, xp, xc, xC, xn = x
        s = xt.shape

        xt = self.token_embed(xt)
        xt = self.token_dropout(xt)

        xp = self.pos_embed(xp)
        xp = self.pos_dropout(xp)

        xc = self.char_embed(xc).mean(dim=2)
        xc = self.char_dropout(xc)

        xC = self.isCap_embed(xC)
        xC = self.isCap_dropout(xC)

        xn = self.isNum_embed(xn)
        xn = self.isNum_dropout(xn)

        x = torch.concat([xt, xp, xc, xC, xn], dim=2)

        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0)
        x = x.contiguous()
        x = x.view(-1, x.shape[2])
        x = self.fc(x)
        return F.log_softmax(x, dim=1).view(s[0], s[1], -1)


class NERNetPretrainedMix(nn.Module):

    def __init__(self, scratch_vocab_size, pretrain_vocab_size, embed_dim, hidden_dim,
                 pretrained_embed, num_class, token_padidx=0, scale_grad_by_freq=False,
                 num_layers=1, dropout=0, proj_size=0, device=None):

        super(NERNetPretrainedMix, self).__init__()

        self.embed_scratch = nn.Embedding(scratch_vocab_size, embed_dim, padding_idx=token_padidx,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)


        self.embed_pretrain = nn.Embedding(pretrain_vocab_size, embed_dim,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)
        if pretrained_embed is not None:
            self.embed_pretrain.from_pretrained(pretrained_embed, freeze=False)


        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout, proj_size=proj_size, device=device)

        """
        Sets forget gate bias
        """
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)

        self.fc = nn.Linear(2*hidden_dim, num_class,
                            device=device) if proj_size == 0 else nn.Linear(2*proj_size, num_class, device=device)

        self.params = nn.ModuleDict({
            'group_1': nn.ModuleList([self.embed_pretrain]),
            'group_2': nn.ModuleList([self.embed_scratch, self.lstm, self.fc])})

    def mix_embedding(self, x):
        pretrain_flag = x >= self.embed_scratch.num_embeddings
        x_scratch = x.clone()
        x_scratch[pretrain_flag] = 0
        x = x - self.embed_scratch.num_embeddings
        x[~pretrain_flag] = 0

        x_scratch = self.embed_scratch(x_scratch)
        x = self.embed_pretrain(x)

        x[~pretrain_flag] = x_scratch[~pretrain_flag]
        return x

    def forward(self, x, x_len):
        x, xp, xc, xC, xn = x
        s = x.shape
        x = self.mix_embedding(x)
        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0)
        x = x.contiguous()
        x = x.view(-1, x.shape[2])
        x = self.fc(x)
        return F.log_softmax(x, dim=1).view(s[0], s[1], -1)

class NERNetAttention(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class, token_padidx=0, scale_grad_by_freq=False,
                 num_layers=1, dropout=0, proj_size=0, device=None):

        super(NERNetAttention, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=token_padidx,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout, proj_size=proj_size, device=device)

        """
        Sets forget gate bias
        """
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)

        self.fc = nn.Linear(2*hidden_dim, num_class,
                            device=device) if proj_size == 0 else nn.Linear(2*proj_size, num_class, device=device)

        self.params = nn.ModuleDict({
            'group_1': nn.ModuleList([self.embedding]),
            'group_2': nn.ModuleList([self.lstm, self.fc])})


    def forward(self, x, x_len):
        x, xp, xc, xC, xn = x
        s = x.shape
        x = self.embedding(x)
        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0)
        """
        attention
        """
        a = torch.bmm(x, x.transpose(2, 1))
        a = F.softmax(a, dim=2)
        x = torch.bmm(a, x)
        """
        FC layer
        """
        x = x.contiguous()
        x = x.view(-1, x.shape[2])
        x = self.fc(x)
        return F.log_softmax(x, dim=1).view(s[0], s[1], -1)


class NERNetBasic(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class,
                 token_padidx=0, pretrained_embed=None, scale_grad_by_freq=False,
                 num_layers=1, dropout=0, proj_size=0, device=None):

        super(NERNetBasic, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=token_padidx,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)
        if pretrained_embed is not None:
            self.embedding.from_pretrained(pretrained_embed, freeze=False)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout, proj_size=proj_size, device=device)

        """
        Sets forget gate bias
        """
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)

        self.fc = nn.Linear(2*hidden_dim, num_class,
                            device=device) if proj_size == 0 else nn.Linear(2*proj_size, num_class, device=device)

        self.params = nn.ModuleDict({
            'group_1': nn.ModuleList([self.embedding]),
            'group_2': nn.ModuleList([self.lstm, self.fc])})


    def forward(self, x, x_len):
        x, xp, xc, xC, xn = x
        s = x.shape
        x = self.embedding(x)
        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0)
        x = x.contiguous()
        x = x.view(-1, x.shape[2])
        x = self.fc(x)
        return F.log_softmax(x, dim=1).view(s[0], s[1], -1)


class NERNetCRF(nn.Module):

    def __init__(self, vocab_size, embed_dim, token_p, hidden_dim, num_class,
                 token_padidx=0, pretrained_embed=None, scale_grad_by_freq=False,
                 num_layers=1, dropout=0, proj_size=0, device=None):

        super(NERNetCRF, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=token_padidx,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)
        if pretrained_embed is not None:
            self.embedding.from_pretrained(pretrained_embed, freeze=False)

        self.token_dropout = nn.Dropout(p=token_p)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout, proj_size=proj_size, device=device)

        """
        Sets forget gate bias
        """
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)

        self.fc = nn.Linear(2*hidden_dim, num_class,
                            device=device) if proj_size == 0 else nn.Linear(2*proj_size, num_class, device=device)

        self.crf = CRF(num_class, batch_first=True).to(device)

        self.params = nn.ModuleDict({
            'group_1': nn.ModuleList([self.embedding]),
            'group_2': nn.ModuleList([self.lstm, self.fc, self.crf])})


    def get_mask(self, x_len, max_len):
        mask = []
        for l in x_len:
            mask.append([1]*l + [0]*(max_len-l))
        return torch.tensor(mask, dtype=torch.bool)

    def run(self, x, x_len):
        x, xp, xc, xC, xn = x
        s = x.shape
        x = self.embedding(x)
        x = self.token_dropout(x)
        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0)
        x = x.contiguous()
        x = x.view(-1, x.shape[2])
        x = self.fc(x)
        x = x.view(s[0], s[1], -1)
        mask = self.get_mask(x_len, x.shape[1]).to(x.device)
        return x, mask

    def forward(self, x, x_len, y):
        x, mask = self.run(x, x_len)
        loss = -self.crf.forward(x, y, mask=mask)
        return self.crf.decode(x, mask), loss

    def decode(self, x, x_len):
        x, mask = self.run(x, x_len)
        return crf.decode(x, mask)




class NERNetCombined(nn.Module):

    def __init__(self, token_embed_params, token_p,
                 char_embed_params, char_p,
                 pos_embed_params, pos_p,
                 isCap_embed_params, isCap_p,
                 isNum_embed_params, isNum_p,
                 hidden_dim, num_class, lstm_p, attn_p,
                 token_padidx=0, pretrained_embed=None, scale_grad_by_freq=False,
                 num_layers=1, dropout=0, proj_size=0, device=None):

        super(NERNetCombined, self).__init__()

        self.token_embed = nn.Embedding(**token_embed_params,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)
        if pretrained_embed is not None:
            self.token_embed.from_pretrained(pretrained_embed, freeze=False)
        self.token_dropout = nn.Dropout(p=token_p)


        self.pos_embed = nn.Embedding(**pos_embed_params,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)
        self.pos_dropout = nn.Dropout(p=pos_p)

        self.char_embed = nn.Embedding(**char_embed_params,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)
        self.char_dropout = nn.Dropout(p=char_p)

        """
        Binary features
        """
        self.isCap_embed = nn.Embedding(**isCap_embed_params,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)
        self.isCap_dropout = nn.Dropout(p=isCap_p)

        self.isNum_embed = nn.Embedding(**isNum_embed_params,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)
        self.isNum_dropout = nn.Dropout(p=isNum_p)



        embed_dim = token_embed_params['embedding_dim'] + \
                    pos_embed_params['embedding_dim'] + \
                    char_embed_params['embedding_dim'] + \
                    isCap_embed_params['embedding_dim'] + \
                    isNum_embed_params['embedding_dim']

        self.layer_norm_embed = nn.LayerNorm(embed_dim, device=device)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout, proj_size=proj_size, device=device)
        self.lstm_dropout = nn.Dropout(p=lstm_p)

        self.layer_norm_lstm = nn.LayerNorm(2*hidden_dim, device=device)

        """
        Sets forget gate bias
        """
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)

        self.attn_dropout = nn.Dropout(p=attn_p)

        self.layer_norm_attn = nn.LayerNorm(2*hidden_dim, device=device)

        self.fc = nn.Linear(2*hidden_dim, num_class,
                            device=device) if proj_size == 0 else nn.Linear(2*proj_size, num_class, device=device)

        self.crf = CRF(num_class, batch_first=True).to(device)

        self.params = nn.ModuleDict({
            'group_1': nn.ModuleList([self.token_embed]),
            'group_2': nn.ModuleList([self.pos_embed, self.char_embed,
                                      self.isCap_embed, self.isNum_embed,
                                      self.lstm, self.fc, self.crf])})

    def freeze_layer(self, group='group_1'):
        for module in self.params[group]:
            for param in module.parameters():
                module.param = False

    def unfreeze_layer(self, group='group_1'):
        for module in self.params[group]:
            for param in module.parameters():
                module.param = True

    def get_mask(self, x_len, max_len):
        mask = []
        for l in x_len:
            mask.append([1]*l + [0]*(max_len-l))
        return torch.tensor(mask, dtype=torch.bool)

    def run(self, x, x_len):
        xt, xp, xc, xC, xn = x
        s = xt.shape

        """
        Different embeddings
        """
        xt = self.token_embed(xt)
        xt = self.token_dropout(xt)

        xp = self.pos_embed(xp)
        xp = self.pos_dropout(xp)

        xc = self.char_embed(xc).mean(dim=2)
        xc = self.char_dropout(xc)

        xC = self.isCap_embed(xC)
        xC = self.isCap_dropout(xC)

        xn = self.isNum_embed(xn)
        xn = self.isNum_dropout(xn)

        x = torch.concat([xt, xp, xc, xC, xn], dim=2)
        x = self.layer_norm_embed(x)

        """
        lstm
        """
        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0)
        x = x.contiguous()
        x = self.layer_norm_lstm(x)

        """
        attention
        """
        a = torch.bmm(x, x.transpose(2, 1))
        a = F.softmax(a, dim=2)
        x = torch.bmm(a, x)
        x = self.layer_norm_attn(x)

        """
        FC layer
        """
        x = x.view(-1, x.shape[2])
        x = self.fc(x)
        x = x.view(s[0], s[1], -1)

        mask = self.get_mask(x_len, x.shape[1]).to(x.device)
        return x, mask

    def forward(self, x, x_len, y):
        x, mask = self.run(x, x_len)
        loss = -self.crf.forward(x, y, mask=mask)
        return self.crf.decode(x, mask), loss

    def decode(self, x, x_len):
        x, mask = self.run(x, x_len)
        return self.crf.decode(x, mask)



class NERNetCombined2(nn.Module):

    def __init__(self, tokenp_embed_params, tokens_embed_params, token_p,
                 char_embed_params, char_p,
                 pos_embed_params, pos_p,
                 isCap_embed_params, isCap_p,
                 isNum_embed_params, isNum_p,
                 hidden_dim, num_class, lstm_p, attn_p,
                 token_padidx=0, pretrained_embed=None, scale_grad_by_freq=False,
                 num_layers=1, dropout=0, proj_size=0, device=None):

        super(NERNetCombined2, self).__init__()

        import pdb; pdb.set_trace()
        self.tokens_embed = nn.Embedding(**tokens_embed_params,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)

        self.tokenp_embed = nn.Embedding(**tokenp_embed_params,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)
        if pretrained_embed is not None:
            self.tokenp_embed.from_pretrained(pretrained_embed, freeze=False)
        self.token_dropout = nn.Dropout(p=token_p)


        self.pos_embed = nn.Embedding(**pos_embed_params,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)
        self.pos_dropout = nn.Dropout(p=pos_p)

        self.char_embed = nn.Embedding(**char_embed_params,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)
        self.char_dropout = nn.Dropout(p=char_p)

        """
        Binary features
        """
        self.isCap_embed = nn.Embedding(**isCap_embed_params,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)
        self.isCap_dropout = nn.Dropout(p=isCap_p)

        self.isNum_embed = nn.Embedding(**isNum_embed_params,
                                     scale_grad_by_freq=scale_grad_by_freq, device=device)
        self.isNum_dropout = nn.Dropout(p=isNum_p)



        embed_dim = token_embed_params['embedding_dim'] + \
                    pos_embed_params['embedding_dim'] + \
                    char_embed_params['embedding_dim'] + \
                    isCap_embed_params['embedding_dim'] + \
                    isNum_embed_params['embedding_dim']

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout, proj_size=proj_size, device=device)
        self.lstm_dropout = nn.Dropout(p=lstm_p)

        """
        Sets forget gate bias
        """
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)

        self.attn_dropout = nn.Dropout(p=attn_p)

        self.fc = nn.Linear(2*hidden_dim, num_class,
                            device=device) if proj_size == 0 else nn.Linear(2*proj_size, num_class, device=device)

        self.crf = CRF(num_class, batch_first=True).to(device)

        self.params = nn.ModuleDict({
            'group_1': nn.ModuleList([self.tokenp_embed]),
            'group_2': nn.ModuleList([self.tokens_embed, self.pos_embed, self.char_embed, self.isCap_embed, self.isNum_embed,
                                      self.lstm, self.fc, self.crf])})

    def freeze_layer(self, group='group_1'):
        for module in self.params[group]:
            module.requires_grad = False

    def unfreeze_layer(self, group='group_1'):
        for module in self.params[group]:
            module.requires_grad = True

    def mix_embedding(self, x):
        pretrain_flag = x >= self.tokens_embed.num_embeddings
        x_scratch = x.clone()
        x_scratch[pretrain_flag] = 0
        x = x - self.tokens_embed.num_embeddings
        x[~pretrain_flag] = 0

        x_scratch = self.tokens_embed(x_scratch)
        x = self.tokenp_embed(x)

        x[~pretrain_flag] = x_scratch[~pretrain_flag]
        return x


    def get_mask(self, x_len, max_len):
        mask = []
        for l in x_len:
            mask.append([1]*l + [0]*(max_len-l))
        return torch.tensor(mask, dtype=torch.bool)


    def run(self, x, x_len):
        xt, xp, xc, xC, xn = x
        s = xt.shape

        """
        Different embeddings
        """
        xt = self.mix_embedding(xt)
        xt = self.token_dropout(xt)

        xp = self.pos_embed(xp)
        xp = self.pos_dropout(xp)

        xc = self.char_embed(xc).mean(dim=2)
        xc = self.char_dropout(xc)

        xC = self.isCap_embed(xC)
        xC = self.isCap_dropout(xC)

        xn = self.isNum_embed(xn)
        xn = self.isNum_dropout(xn)

        x = torch.concat([xt, xp, xc, xC, xn], dim=2)

        """
        lstm
        """
        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0)
        x = x.contiguous()

        """
        attention
        """
        a = torch.bmm(x, x.transpose(2, 1))
        a = F.softmax(a, dim=2)
        x = torch.bmm(a, x)

        """
        FC layer
        """
        x = x.view(-1, x.shape[2])
        x = self.fc(x)
        x = x.view(s[0], s[1], -1)

        mask = self.get_mask(x_len, x.shape[1]).to(x.device)
        return x, mask

    def forward(self, x, x_len, y):
        x, mask = self.run(x, x_len)
        loss = -self.crf.forward(x, y, mask=mask)
        return self.crf.decode(x, mask), loss

    def decode(self, x, x_len):
        x, mask = self.run(x, x_len)
        return self.crf.decode(x, mask)

