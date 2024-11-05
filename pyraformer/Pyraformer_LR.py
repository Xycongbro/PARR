import sys

import numpy as np
import torch
import torch.nn as nn
from .Layers import EncoderLayer, Predictor, Decoder, DecoderLayer_Mul
from .Layers import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct
from .Layers import get_mask, get_subsequent_mask, refer_points, get_k_q, get_q_k
from .embed import DataEmbedding, CustomEmbedding, DataEmbedding2
from .Predictor import FeatureProcessingModule
from .LSTM import LSTMEncoder
class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, opt):
        super().__init__()

        self.d_model = opt.d_model
        self.model_type = opt.model
        self.window_size = opt.window_size
        self.truncate = opt.truncate
        if opt.decoder == 'attention':
            self.mask, self.all_size = get_mask(opt.input_size, opt.window_size, opt.inner_size, opt.device)
            self.indexes = refer_points(self.all_size, opt.window_size, opt.device)
        else:
            # self.mask, self.all_size = get_mask(opt.input_size + 1, opt.window_size, opt.inner_size, opt.device)
            self.mask, self.all_size = get_mask(opt.input_size + opt.predict_step, opt.window_size, opt.inner_size,
                                                opt.device)

        self.decoder_type = opt.decoder
        if opt.decoder == 'FC':
            self.indexes = refer_points(self.all_size, opt.window_size, opt.device)

        if opt.use_tvm:
            assert len(set(self.window_size)) == 1, "Only constant window size is supported."
            padding = 1 if opt.decoder == 'FC' else 0
            q_k_mask = get_q_k(opt.input_size + padding, opt.inner_size, opt.window_size[0], opt.device)
            k_q_mask = get_k_q(q_k_mask)
            self.layers = nn.ModuleList([
                EncoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout,
                             normalize_before=False, use_tvm=True, q_k_mask=q_k_mask, k_q_mask=k_q_mask) for i in
                range(opt.n_layer)
            ])
        else:
            self.layers = nn.ModuleList([
                EncoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout,
                             normalize_before=False) for i in range(opt.n_layer)
            ])

        if opt.embed_type == 'CustomEmbedding':
            self.enc_embedding = CustomEmbedding(opt.enc_in, opt.d_model, opt.covariate_size, opt.seq_num, opt.dropout)
        else:
            self.enc_embedding = DataEmbedding(opt.enc_in, opt.sta_in, opt.d_model, opt.dropout, opt.input_size + opt.predict_step)

        self.conv_layers = eval(opt.CSCM)(opt.d_model, opt.window_size, opt.d_bottleneck)
        self.layer_norm = nn.LayerNorm(opt.d_model)
    def forward(self, x_enc, x_mark_enc):
        # 交换x_enc的行
        #
        # new_order = [0, 1, 15, 2, 3, 16, 4, 5, 17, 6, 7, 18, 8, 9, 19, 10, 11, 20, 12, 13, 14, 21]
        # 使用新的索引顺序重新排列 x_enc
        # x_enc_reordered = x_enc[:, new_order]
        seq_enc = self.enc_embedding(x_enc, x_mark_enc)
        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        seq_enc = self.conv_layers(seq_enc)
        # all_enc_list = []
        for i in range(len(self.layers)):
            seq_enc_residual = seq_enc
            seq_enc, _ = self.layers[i](seq_enc, mask)
            seq_enc = self.layer_norm(seq_enc + seq_enc_residual)  # 残差连接和层规范化

            # all_enc_list.append(seq_enc)
        # all_seq_enc = torch.stack(all_enc_list, dim=2)
        # return seq_enc
        if self.decoder_type == 'FC':
            indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
            indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
            all_enc = torch.gather(seq_enc, 1, indexes)
            seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)

            # indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
            # all_enc = torch.gather(all_seq_enc, 1, indexes)
            # seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)
        elif self.decoder_type == 'attention' and self.truncate:
            seq_enc = seq_enc[:, :self.all_size[0]]
        return seq_enc


class Decoder_Mul(nn.Module):
    """ A decoder model with self attention mechanism. """

    def __init__(self, opt):
        super().__init__()

        self.d_model = opt.d_model
        self.model_type = opt.model
        self.window_size = opt.window_size
        self.truncate = opt.truncate
        if opt.decoder == 'attention':
            self.mask, self.all_size = get_mask(opt.input_size, opt.window_size, opt.inner_size, opt.device)
            self.indexes = refer_points(self.all_size, opt.window_size, opt.device)
        else:
            # self.mask, self.all_size = get_mask(opt.input_size + 1, opt.window_size, opt.inner_size, opt.device)
            self.mask, self.all_size = get_mask(opt.input_size + opt.predict_step, opt.window_size, opt.inner_size,
                                                opt.device)

        self.decoder_type = opt.decoder
        if opt.decoder == 'FC':
            self.indexes = refer_points(self.all_size, opt.window_size, opt.device)

        if opt.use_tvm:
            assert len(set(self.window_size)) == 1, "Only constant window size is supported."
            padding = 1 if opt.decoder == 'FC' else 0
            q_k_mask = get_q_k(opt.input_size + padding, opt.inner_size, opt.window_size[0], opt.device)
            k_q_mask = get_k_q(q_k_mask)
            self.layers = nn.ModuleList([
                DecoderLayer_Mul(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout,
                                 normalize_before=False, use_tvm=True, q_k_mask=q_k_mask, k_q_mask=k_q_mask) for i in
                range(opt.n_layer)
            ])
        else:
            self.layers = nn.ModuleList([
                DecoderLayer_Mul(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout,
                                 normalize_before=False) for i in range(opt.n_layer)
            ])

        if opt.embed_type == 'CustomEmbedding':
            self.enc_embedding = CustomEmbedding(opt.enc_in, opt.d_model, opt.covariate_size, opt.seq_num, opt.dropout)
        else:
            self.enc_embedding = DataEmbedding2(opt.enc_in, opt.sta_in, opt.d_model, opt.dropout)

        self.conv_layers = eval(opt.CSCM)(opt.d_model, opt.window_size, opt.d_bottleneck)

    def forward(self, x_enc, x_mark_enc, enc_output):
        # 交换x_enc的行

        # 使用新的索引顺序重新排列 x_enc
        # x_enc_reordered = x_enc[:, new_order]
        seq_enc = self.enc_embedding(x_enc, x_mark_enc)
        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        seq_enc = self.conv_layers(seq_enc)
        # all_enc_list = []
        for i in range(len(self.layers)):
            seq_enc, _, _ = self.layers[i](seq_enc, enc_output, mask, mask)
            # all_enc_list.append(seq_enc)
        # all_seq_enc = torch.stack(all_enc_list, dim=2)
        # return seq_enc[:, :37, :]
        if self.decoder_type == 'FC':
            indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
            indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
            all_enc = torch.gather(seq_enc, 1, indexes)
            seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)

            # indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
            # all_enc = torch.gather(all_seq_enc, 1, indexes)
            # seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)
        elif self.decoder_type == 'attention' and self.truncate:
            seq_enc = seq_enc[:, :self.all_size[0]]
        return seq_enc


class Model(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, opt):
        super().__init__()

        self.predict_step = opt.predict_step
        self.d_model = opt.d_model
        self.input_size = opt.input_size
        self.decoder_type = opt.decoder
        self.channels = opt.enc_in
        self.encoder = Encoder(opt)
        if opt.decoder == 'attention':
            mask = get_subsequent_mask(opt.input_size, opt.window_size, opt.predict_step, opt.truncate)
            self.decoder = Decoder(opt, mask)
            # self.predictor = Predictor(opt.d_model, 1)
            # self.predictor = Predictor(opt.d_model, opt.enc_in)
        elif opt.decoder == 'FC':
            # self.predictor = FeatureProcessingModule(4 * opt.d_model, opt.dropout)
            # self.decoder = Decoder_Mul(opt)
            # self.fc = nn.Linear(4 * opt.d_model, opt.d_model)
            # self.decoder = LSTMEncoder(4 * opt.d_model, opt.d_model, opt.dropout)
            # self.decoder = Predictor(opt.d_model, opt.d_model // 4, opt.dropout, 1)
            self.predictor = Predictor(4 * opt.d_model, opt.d_model, opt.dropout, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, pretrain):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        # x_enc_cpu = x_enc.cpu()
        # x_dec_cpu = x_dec.cpu()
        # combined_enc = np.zeros_like(x_enc_cpu)
        # combined_enc[:, :23, :-1] = x_enc_cpu[:, 7:30, :-1]
        # combined_enc[:, 23:30, :-1] = x_dec_cpu[:, :, :-1]
        # combined_enc[:, :, -1] = x_dec_cpu[:, :, -1]

        # x_enc_cpu = x_enc.cpu()
        # x_dec_cpu = x_dec.cpu()
        # combined_enc = np.zeros_like(x_enc_cpu)
        # combined_enc[:, :8, :-1] = x_enc_cpu[:, 7:15, :-1]
        # combined_enc[:, 8:15, :-1] = x_dec_cpu[:, :, :-1]
        # combined_enc[:, :, -1] = x_enc_cpu[:, :, -1]

        # print(combined_enc[0])
        # sys.exit(0)

        # x_enc = torch.tensor(combined_enc, device=x_enc.device)
        if self.decoder_type == 'attention':
            enc_output = self.encoder(x_enc, x_mark_enc)
            dec_enc = self.decoder(x_dec, x_mark_dec, enc_output)
            if pretrain:
                dec_enc = torch.cat([enc_output[:, :self.input_size], dec_enc], dim=1)
                pred = self.predictor(dec_enc)
            else:
                pred = self.predictor(dec_enc)
        elif self.decoder_type == 'FC':
            # enc_output = self.encoder(x_enc, x_mark_enc)[:, -1, :]
            # new_order = [2, 5, 8, 11, 14, 17, 21]

            # enc_output = self.encoder(x_enc, x_mark_enc)[:, -7:, :]
            #
            # # enc_output = self.encoder(x_enc, x_mark_enc)[:, -7:, :]
            #
            # # enc_output = self.fc(enc_output)
            # # print(self.predictor(enc_output).shape)
            # pred = self.predictor(enc_output).view(enc_output.size(0), self.predict_step, -1)
            enc_output = self.encoder(x_enc, x_mark_enc)
            enc_output = enc_output[:, -7:, :]
            # enc_output = self.encoder(x_enc, x_mark_enc)
            # dec_output = self.decoder(x_enc, x_mark_dec, enc_output)[:, -7:, :]
            # enc_output = self.fc(enc_output)

            # enc_output = self.decoder(enc_output, x_mark_enc)
            # pred = enc_output
            #
            pred = self.predictor(enc_output).view(enc_output.size(0), self.predict_step, -1)
            # pred = self.predictor(dec_output).view(dec_output.size(0), self.predict_step, -1)
        return pred
