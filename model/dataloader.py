import torch
from torch import nn
import torchtext.vocab
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def pad_collate(batch, padding_values):

    if len(padding_values) == 2:
        (xt_pad, xp_pad, xc_pad), y_pad = padding_values
        xxt, xxp, xxc, yy, mm = zip(*batch)

        y_len = [len(y) for y in yy]
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=y_pad)
    else:
        (xt_pad, xp_pad, xc_pad), = padding_values
        xxt, xxp, xxc, mm = zip(*batch)

    x_len = [len(x) for x in xxt]

    xxt_pad = pad_sequence(xxt, batch_first=True, padding_value=xt_pad)
    xxp_pad = pad_sequence(xxp, batch_first=True, padding_value=xp_pad)

    #xxc_rmax, xxc_cmax = 0, 0
    #for xx in xxc:
    #    if xxc_rmax < xx.shape[0]:
    #        xxc_rmax = xx.shape[0]
    #    if xxc_cmax < xx.shape[1]:
    #        xxc_cmax = xx.shape[1]

    # xxc_pad = []
    # for xx in xxc:
    #     m = nn.ConstantPad2d((0, xxc_cmax-xx.shape[1], 0, xxc_rmax-xx.shape[0]), xc_pad)
    #     xxc_pad.append(m(xx)[None])
    # xxc_pad = torch.concat(xxc_pad)
    xxc_pad = None

    if len(padding_values) == 2:
        return (xxt_pad, xxp_pad, xxc_pad), yy_pad, x_len, y_len, mm
    else:
        return (xxt_pad, xxp_pad, xxc_pad), x_len, mm


class NERDataset(Dataset):

    def __init__(self, data, vocab, pad_token, device=None, is_predict=False):
        self.data = data
        self.vocab = vocab
        self.device = device
        self.pad_token = pad_token
        self.is_predict = is_predict

        self.data_tensors = list(map(lambda x:torch.tensor(vocab['token'](x)).to(device), data['token']))
        self.pos_tensors = list(map(lambda x:torch.tensor(vocab['pos'](x)).to(device), data['pos']))
        if not is_predict:
            self.tag_tensors = list(map(lambda x:torch.tensor(vocab['tag'](x)).to(device), data['tag']))

    def __len__(self):
        return len(self.data['token'])

    def __getitem__(self, idx):
        """
        token = torch.tensor(self.vocab['token'](self.data['token'][idx]), device=self.device)
        pos = torch.tensor(self.vocab['pos'](self.data['pos'][idx]), device=self.device)
        c = [torch.tensor(self.vocab['char'](chars), device=self.device) for chars in self.data['char'][idx]]
        char = pad_sequence(c, batch_first=True, padding_value=self.vocab['char'][self.pad_token])
        """
        char = None
        mask = self.data['mask'][idx]

        token = self.data_tensors[idx]
        pos = self.pos_tensors[idx]
        if not self.is_predict:
            """
            tag = torch.tensor(self.vocab['tag'](self.data['tag'][idx]), device=self.device)
            """
            tag = self.tag_tensors[idx]
            return token, pos, char, tag, mask
        else:
            return token, pos, char, mask


def pad_collate_char(batch, padding_values):

    if len(padding_values) == 2:
        (xt_pad, xp_pad, xc_pad, xC_pad, xn_pad), y_pad = padding_values
        xxt, xxp, xxc, xxC, xxn, yy, mm = zip(*batch)

        y_len = [len(y) for y in yy]
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=y_pad)
    else:
        (xt_pad, xp_pad, xc_pad, xC_pad, xn_pad), = padding_values
        xxt, xxp, xxc, xxC, xxn, mm = zip(*batch)

    x_len = [len(x) for x in xxt]

    xxt_pad = pad_sequence(xxt, batch_first=True, padding_value=xt_pad)
    xxp_pad = pad_sequence(xxp, batch_first=True, padding_value=xp_pad)
    xxC_pad = pad_sequence(xxC, batch_first=True, padding_value=xC_pad)
    xxn_pad = pad_sequence(xxn, batch_first=True, padding_value=xn_pad)

    xxc_rmax, xxc_cmax = 0, 0
    for xx in xxc:
        if xxc_rmax < xx.shape[0]:
            xxc_rmax = xx.shape[0]
        if xxc_cmax < xx.shape[1]:
            xxc_cmax = xx.shape[1]

    xxc_pad = []
    for xx in xxc:
        m = nn.ConstantPad2d((0, xxc_cmax-xx.shape[1], 0, xxc_rmax-xx.shape[0]), xc_pad)
        xxc_pad.append(m(xx)[None])
    xxc_pad = torch.concat(xxc_pad)

    if len(padding_values) == 2:
        return (xxt_pad, xxp_pad, xxc_pad, xxC_pad, xxn_pad), yy_pad, x_len, y_len, mm
    else:
        return (xxt_pad, xxp_pad, xxc_pad, xxC_pad, xxn_pad), x_len, mm


class NERDatasetChar(Dataset):

    def __init__(self, data, vocab, pad_token, device=None, is_predict=False):
        self.data = data
        self.vocab = vocab
        self.device = device
        self.pad_token = pad_token
        self.is_predict = is_predict

        self.data_tensors = list(map(lambda x:torch.tensor(vocab['token'](x)).to(device), data['token']))
        self.pos_tensors = list(map(lambda x:torch.tensor(vocab['pos'](x)).to(device), data['pos']))

        pad_char = lambda c :pad_sequence(c, batch_first=True, padding_value=self.vocab['char'][self.pad_token])
        self.char_tensors = list(map(lambda x:pad_char([torch.tensor(self.vocab['char'](chars), device=self.device) for chars in x]), data['char'] ))

        self.isNumber_tensors = list(map(lambda x:torch.tensor(x).to(device), data['isNumber']))
        self.isCapital_tensor = list(map(lambda x:torch.tensor(x).to(device), data['isCapital']))

        if not is_predict:
            self.tag_tensors = list(map(lambda x:torch.tensor(vocab['tag'](x)).to(device), data['tag']))

    def __len__(self):
        return len(self.data['token'])

    def __getitem__(self, idx):
        """
        token = torch.tensor(self.vocab['token'](self.data['token'][idx]), device=self.device)
        pos = torch.tensor(self.vocab['pos'](self.data['pos'][idx]), device=self.device)
        c = [torch.tensor(self.vocab['char'](chars), device=self.device) for chars in self.data['char'][idx]]
        char = pad_sequence(c, batch_first=True, padding_value=self.vocab['char'][self.pad_token])
        """

        token = self.data_tensors[idx]
        pos = self.pos_tensors[idx]
        char = self.char_tensors[idx]

        isCap = self.isCapital_tensor[idx]
        isNum = self.isNumber_tensors[idx]

        mask = self.data['mask'][idx]

        if not self.is_predict:
            """
            tag = torch.tensor(self.vocab['tag'](self.data['tag'][idx]), device=self.device)
            """
            tag = self.tag_tensors[idx]
            return token, pos, char, isCap, isNum, tag, mask
        else:
            return token, pos, char, isCap, isNum, mask

