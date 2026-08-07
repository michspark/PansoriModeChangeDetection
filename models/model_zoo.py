import torch
import torch.nn as nn
from torchaudio.models import Conformer
from .model_utils import calc_conv_param
from .modules import Conv2DBlock

class Conv2DGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.params = calc_conv_param(config)

        self.kernel_size = config.kernel_size
        self.dilation = config.dilation
        self.num_classes = config.num_classes
        self.enc = self.build_enc()

        self.hidden_dim = config.hidden_dim
        self.num_gru = config.num_gru
        self.dropout = config.dropout

        if config.pool_size:
            out_freq = config.num_bins // (2 ** config.num_layers)
        else:
            out_freq = config.num_bins

        gru_input = self.params[-1]['output_channel'] * out_freq
        self.gru = nn.GRU(input_size=gru_input, hidden_size=self.hidden_dim,
                          num_layers=self.num_gru, batch_first=True,
                          dropout=self.dropout, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim * 2, self.num_classes)

    def build_enc(self):
        enc = nn.Sequential()
        for idx, param in enumerate(self.params):
            enc.add_module(f'conv_{idx}',
                Conv2DBlock(param['input_channel'], param['output_channel'],
                            kernel_size=self.kernel_size, padding='same',
                            dilation=tuple(self.dilation)))
            if self.config.cnn_dropout:
                enc.add_module(f'dropout_{idx}', nn.Dropout2d(self.config.cnn_dropout))
            if self.config.pool_size:
                enc.add_module(f'pool_{idx}', nn.MaxPool2d(tuple(param['max_pool'])))
        return enc

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.enc(x)
        b, _, _, t = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, t, -1)

        x, _ = self.gru(x)
        x = self.fc(x)

        return x

class Conv2DConformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.params = calc_conv_param(config)

        self.kernel_size = config.kernel_size
        self.dilation = config.dilation
        self.num_classes = config.num_classes
        self.enc = self.build_enc()

        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout

        if config.pool_size :
            out_freq = config.num_bins // (2 ** config.num_layers)
        else:
            out_freq = config.num_bins

        cnn_out_dim = self.params[-1]['output_channel'] * out_freq

        self.conformer = Conformer(
            input_dim=cnn_out_dim,
            num_heads=4,
            ffn_dim=cnn_out_dim * 4,
            num_layers=4,
            depthwise_conv_kernel_size=31,
            dropout=self.dropout
        )

        self.fc = nn.Conv1d(cnn_out_dim, self.num_classes, kernel_size=5, padding=2)

    def build_enc(self):
        enc = nn.Sequential()
        for idx, param in enumerate(self.params):
            enc.add_module(f'conv_{idx}',
                Conv2DBlock(param['input_channel'], param['output_channel'],
                            kernel_size=self.kernel_size, padding='same',
                            dilation=tuple(self.dilation)))
            if self.config.cnn_dropout:
                enc.add_module(f'dropout_{idx}', nn.Dropout2d(self.config.cnn_dropout))
            if self.config.pool_size:
                enc.add_module(f'pool_{idx}', nn.MaxPool2d(tuple(param['max_pool'])))
        return enc

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.enc(x)
        b, _, _, t = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, t, -1)

        lengths = torch.full((b,), t, dtype=torch.long, device=x.device)
        x, _ = self.conformer(x, lengths)

        x = x.permute(0, 2, 1)   # (Batch, Feature, Time)
        x = self.fc(x)           # (Batch, num_classes, Time)
        x = x.permute(0, 2, 1)   # Loss 계산을 위해 원래대로 복구: (Batch, Time, num_classes)

        return x


class Conv2DTCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.params = calc_conv_param(config)

        self.kernel_size = config.kernel_size
        self.dilation = config.dilation
        self.num_classes = config.num_classes
        self.enc = self.build_enc()

        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout

        # CNN을 통과하고 나온 피쳐 차원 크기 계산
        if config.pool_size:
            out_freq = config.num_bins // (2 ** config.num_layers)
        else:
            out_freq = config.num_bins

        cnn_out_dim = self.params[-1]['output_channel'] * out_freq

        num_tcn_layers = 6
        tcn_channels = [self.hidden_dim] * num_tcn_layers

        self.tcn = TemporalConvNet(
            num_inputs=cnn_out_dim,
            num_channels=tcn_channels,
            kernel_size=3,  # 시간축 커널 사이즈 (반드시 홀수: 3, 5 등)
            dropout=self.dropout
        )
        # ----------------------------------------------------
        self.fc = nn.Conv1d(self.hidden_dim, self.num_classes, kernel_size=5, padding=2)

    def build_enc(self):
        enc = nn.Sequential()
        for idx, param in enumerate(self.params):
            enc.add_module(f'conv_{idx}',
                Conv2DBlock(param['input_channel'], param['output_channel'],
                            kernel_size=self.kernel_size, padding='same',
                            dilation=tuple(self.dilation)))
            if self.config.cnn_dropout:
                enc.add_module(f'dropout_{idx}', nn.Dropout2d(self.config.cnn_dropout))
            if self.config.pool_size:
                enc.add_module(f'pool_{idx}', nn.MaxPool2d(tuple(param['max_pool'])))
        return enc

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # 1. CNN 인코더 통과
        x = self.enc(x)
        b, _, _, t = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, t, -1)  # (Batch, Time, Feature)

        # ----------------------------------------------------
        # ★ 수정 3: 차원 변경 후 TCN 통과

        x = x.permute(0, 2, 1)   # 1D Conv를 위해 차원 변경: (Batch, Feature, Time)

        x = self.tcn(x)          # 양방향 TCN 통과. 형태 유지: (Batch, Hidden, Time)

        x = self.fc(x)           # Classifier 통과. (Batch, num_classes, Time)

        x = x.permute(0, 2, 1)   # Loss 계산을 위해 원래대로 복구: (Batch, Time, num_classes)
        # ----------------------------------------------------

        return x