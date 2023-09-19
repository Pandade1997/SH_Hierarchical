# -*- coding: utf-8 -*-
import torch
from torch import nn
import numpy as np


class LayerNorm(nn.Module):
    def __init__(self, c, f):
        super(LayerNorm, self).__init__()
        self.ln = nn.LayerNorm([c, f])

    def forward(self, x):
        x = self.ln(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x


class Encoder(nn.Module):

    def __init__(self, in_channel=32):
        super(Encoder, self).__init__()
        self.in_channel = in_channel
        self.conv_1 = nn.Conv2d(self.in_channel, 32, kernel_size=(2, 5), stride=(1, 2), padding=(1, 1))
        # self.conv_1 = nn.Conv2d(18, 32, kernel_size=(2, 5), stride=(1, 2), padding=(1, 1))
        self.bn_1 = nn.BatchNorm2d(32, eps=1e-8)
        self.act_1 = nn.PReLU(32)

        self.conv_2 = nn.Conv2d(32, 32, kernel_size=(2, 3), stride=(1, 2), padding=(1, 1))
        self.bn_2 = nn.BatchNorm2d(32, eps=1e-8)
        self.act_2 = nn.PReLU(32)

        self.conv_3 = nn.Conv2d(32, 32, kernel_size=(2, 3), stride=(1, 1), padding=(1, 1))
        self.bn_3 = nn.BatchNorm2d(32, eps=1e-8)
        self.act_3 = nn.PReLU(32)

        self.conv_4 = nn.Conv2d(32, 64, kernel_size=(2, 3), stride=(1, 1), padding=(1, 1))
        self.bn_4 = nn.BatchNorm2d(64, eps=1e-8)
        self.act_4 = nn.PReLU(64)

        self.conv_5 = nn.Conv2d(64, 128, kernel_size=(2, 3), stride=(1, 1), padding=(1, 1))
        self.bn_5 = nn.BatchNorm2d(128, eps=1e-8)
        self.act_5 = nn.PReLU(128)

    def forward(self, x):
        x_1 = self.act_1(self.bn_1(self.conv_1(x)[:, :, :-1, :]))
        x_2 = self.act_2(self.bn_2(self.conv_2(x_1)[:, :, :-1, :]))
        x_3 = self.act_3(self.bn_3(self.conv_3(x_2)[:, :, :-1, :]))
        x_4 = self.act_4(self.bn_4(self.conv_4(x_3)[:, :, :-1, :]))
        x_5 = self.act_5(self.bn_5(self.conv_5(x_4)[:, :, :-1, :]))
        return [x_1, x_2, x_3, x_4, x_5]


class Decoder(nn.Module):
    def __init__(self, out_channel=32):
        super(Decoder, self).__init__()
        self.out_channel = out_channel
        self.imag_dconv_1 = nn.ConvTranspose2d(128 * 2, 64, kernel_size=(2, 3), stride=(1, 1))
        self.imag_bn_1 = nn.BatchNorm2d(64, eps=1e-8)
        self.imag_act_1 = nn.PReLU(64)

        self.imag_dconv_2 = nn.ConvTranspose2d(128, 32, kernel_size=(2, 3), stride=(1, 1))
        self.imag_bn_2 = nn.BatchNorm2d(32, eps=1e-8)
        self.imag_act_2 = nn.PReLU(32)

        self.imag_dconv_3 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 3), stride=(1, 1))
        self.imag_bn_3 = nn.BatchNorm2d(32, eps=1e-8)
        self.imag_act_3 = nn.PReLU(32)

        self.imag_dconv_4 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 3), stride=(1, 2))
        self.imag_bn_4 = nn.BatchNorm2d(32, eps=1e-8)
        self.imag_act_4 = nn.PReLU(32)

        # self.imag_dconv_5 = nn.ConvTranspose2d(64, 2, kernel_size=(2, 5), stride=(1, 2))
        # self.imag_bn_5 = nn.BatchNorm2d(2, eps=1e-8)
        # self.imag_act_5 = nn.PReLU(2)
        self.imag_dconv_5 = nn.ConvTranspose2d(64, self.out_channel, kernel_size=(2, 5), stride=(1, 2))
        self.imag_bn_5 = nn.BatchNorm2d(self.out_channel, eps=1e-8)
        self.imag_act_5 = nn.PReLU(self.out_channel)

    def forward(self, dprnn_out, encoder_out):
        skipcon_1 = torch.cat([encoder_out[4], dprnn_out], 1)
        x_1 = self.imag_act_1(self.imag_bn_1(self.imag_dconv_1(skipcon_1)[:, :, :-1, 1:-1]))
        skipcon_2 = torch.cat([encoder_out[3], x_1], 1)
        x_2 = self.imag_act_2(self.imag_bn_2(self.imag_dconv_2(skipcon_2)[:, :, :-1, 1:-1]))
        skipcon_3 = torch.cat([encoder_out[2], x_2], 1)
        x_3 = self.imag_act_3(self.imag_bn_3(self.imag_dconv_3(skipcon_3)[:, :, :-1, 1:-1]))
        skipcon_4 = torch.cat([encoder_out[1], x_3], 1)
        x_4 = self.imag_act_4(self.imag_bn_4(self.imag_dconv_4(skipcon_4)[:, :, :-1, :-1]))
        skipcon_5 = torch.cat([encoder_out[0], x_4], 1)
        x_5 = self.imag_dconv_5(skipcon_5)[:, :, :-1, 1:-1]

        return x_5


class DPRNN(nn.Module):
    def __init__(self, numUnits=128, width=50, channel=128, **kwargs):
        super(DPRNN, self).__init__(**kwargs)
        self.numUnits = numUnits
        self.intra_rnn = nn.LSTM(input_size=self.numUnits, hidden_size=int(self.numUnits // 2), batch_first=True,
                                 bidirectional=True)
        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)
        self.intra_ln = nn.LayerNorm([50, 128])
        self.inter_rnn = nn.LSTM(input_size=self.numUnits, hidden_size=self.numUnits, batch_first=True,
                                 bidirectional=False)
        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)
        self.inter_ln = nn.LayerNorm([50, 128])

        self.width = width
        self.channel = channel

    def forward(self, x):
        # x.shape = (Bs, C, T, F)
        self.intra_rnn.flatten_parameters()
        self.inter_rnn.flatten_parameters()
        x = x.permute(0, 2, 3, 1)  # (Bs, T, F, C)
        if not x.is_contiguous():
            x = x.contiguous()
        ## Intra RNN
        # import pdb; pdb.set_trace()
        intra_LSTM_input = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (Bs*T, F, C)
        intra_LSTM_out = self.intra_rnn(intra_LSTM_input)[0]  # (Bs*T, F, C)
        intra_dense_out = self.intra_fc(intra_LSTM_out)
        intra_ln_input = intra_dense_out.view(x.shape[0], -1, self.width, self.channel)  # (Bs, T, F, C)
        intra_out = self.intra_ln(intra_ln_input)
        intra_out = torch.add(x, intra_out)
        ## Inter RNN
        inter_LSTM_input = intra_out.permute(0, 2, 1, 3)  # (Bs, F, T, C)
        inter_LSTM_input = inter_LSTM_input.contiguous()
        inter_LSTM_input = inter_LSTM_input.view(inter_LSTM_input.shape[0] * inter_LSTM_input.shape[1],
                                                 inter_LSTM_input.shape[2], inter_LSTM_input.shape[3])  # (Bs * F, T, C)
        inter_LSTM_out = self.inter_rnn(inter_LSTM_input)[0]
        inter_dense_out = self.inter_fc(inter_LSTM_out)
        inter_dense_out = inter_dense_out.view(x.shape[0], self.width, -1, self.channel)  # (Bs, F, T, C)
        inter_ln_input = inter_dense_out.permute(0, 2, 1, 3)  # (Bs, T, F, C)
        inter_out = self.inter_ln(inter_ln_input)

        #  inter_out = inter_out.permute(0,3,1,2) #(Bs, T, F, C)
        inter_out = torch.add(intra_out, inter_out)
        inter_out = inter_out.permute(0, 3, 1, 2)
        inter_out = inter_out.contiguous()

        return inter_out


class DPCRN_NET(nn.Module):
    # autoencoder = True
    def __init__(self, in_channel=32, out_channel=32):
        super(DPCRN_NET, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.encoder = Encoder(in_channel=self.in_channel)
        self.dprnn1 = DPRNN(128, 50, 128)
        self.dprnn2 = DPRNN(128, 50, 128)
        self.decoder = Decoder(out_channel=self.out_channel)
        self.n_fft = 400
        self.hop_length = 200

    def forward(self, x):
        # x : b, mic, wave

        b, m, t = x.shape[0], x.shape[1], x.shape[2],
        x = x.reshape(-1, t)
        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, center=True)
        # X: bm F T ri 
        F, T = X.shape[1], X.shape[2]
        X = X.reshape(b, m, F, T, 2)
        X0 = torch.cat([X[..., 0], X[..., 1]], dim=1)
        # x:[batch, channel, frequency, time]

        X0 = X0.permute(0, 1, 3, 2)  # (Bs, c, T, F)
        encoder_out = self.encoder(X0)

        dprnn_out = self.dprnn1(encoder_out[4])
        dprnn_out = self.dprnn2(dprnn_out)
        Y = self.decoder(dprnn_out, encoder_out)
        Y = Y.permute(0, 1, 3, 2)
        #  import pdb; pdb.set_trace()

        m_out = int(Y.shape[1] // 2)
        Y_r = Y[:, :m_out]
        Y_i = Y[:, m_out:]
        Y = torch.stack([Y_r, Y_i], dim=-1)
        Y = Y.reshape(-1, F, T, 2)

        y = torch.istft(Y, n_fft=self.n_fft, hop_length=self.hop_length, center=True, length=t)
        y = y.reshape(b, m_out, y.shape[-1]).squeeze(1)

        return y


class DPCRN_multi(nn.Module):
    # autoencoder = True
    def __init__(self):
        super(DPCRN_multi, self).__init__()

        self.DPCRN_1 = DPCRN_NET(in_channel=(1 + 1) ** 2 * 2, out_channel=(1 + 1) ** 2 * 2)
        self.DPCRN_2 = DPCRN_NET(in_channel=(2 + 1) ** 2 * 2, out_channel=(2 + 1) ** 2 * 2)
        self.DPCRN_3 = DPCRN_NET(in_channel=(3 + 1) ** 2 * 2, out_channel=(3 + 1) ** 2 * 2)
        self.DPCRN_4 = DPCRN_NET(in_channel=(4 + 1) ** 2 * 2, out_channel=(4 + 1) ** 2 * 2)

    def forward(self, input_1, input_2, input_3, input_4):
        # x : b, mic, wave

        out_1 = self.DPCRN_1(input_1)

        input_2 = torch.cat([out_1, input_2], dim=1)
        out_2 = self.DPCRN_2(input_2)

        input_3 = torch.cat([out_2, input_3], dim=1)
        out_3 = self.DPCRN_3(input_3)

        input_4 = torch.cat([out_3, input_4], dim=1)
        out_4 = self.DPCRN_4(input_4)

        return out_1, out_2, out_3, out_4


if __name__ == '__main__':
    # complexity()
    from thop import profile, clever_format
    from torch.autograd import Variable

    input_1 = Variable(torch.FloatTensor(torch.rand(1, 4, 16000)))
    input_2 = Variable(torch.FloatTensor(torch.rand(1, 5, 16000)))
    input_3 = Variable(torch.FloatTensor(torch.rand(1, 7, 16000)))
    input_4 = Variable(torch.FloatTensor(torch.rand(1, 9, 16000)))

    net = DPCRN_multi()
    macs, params = profile(net, inputs=(input_1, input_2, input_3, input_4,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)

# 14.056G
# 3.316M
# 这是论文里用到的网络结构
