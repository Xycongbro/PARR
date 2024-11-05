import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.decoder = Decoder()
        self.encoder = Encoder()
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, pretrain):
        print(x_enc.shape)
