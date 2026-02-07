import math
import torch
from torch import nn
import torch.nn.functional as F
from utils.log import logger


class Conv2dSame(nn.Module):
    """2D convolution with TensorFlow-style 'same' padding."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0,
            bias=bias,
        )

    def _pad_same(self, x):
        in_h, in_w = x.shape[-2:]
        k_h, k_w = self.conv.kernel_size
        s_h, s_w = self.conv.stride
        d_h, d_w = self.conv.dilation

        out_h = math.ceil(in_h / s_h)
        out_w = math.ceil(in_w / s_w)

        pad_h = max((out_h - 1) * s_h + (k_h - 1) * d_h + 1 - in_h, 0)
        pad_w = max((out_w - 1) * s_w + (k_w - 1) * d_w + 1 - in_w, 0)

        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

        return x

    def forward(self, x):
        x = self._pad_same(x)
        return self.conv(x)


class ChronoBlock2D(nn.Module):
    """Inception-style 2D conv block used in ChronoNet."""

    def __init__(self, in_channels, out_channels, kernel_sizes=(2, 4, 8), stride=2):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                Conv2dSame(
                    in_channels,
                    out_channels,
                    kernel_size=ks,
                    stride=stride,
                )
                for ks in kernel_sizes
            ]
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        outputs = [self.relu(conv(x)) for conv in self.convs]
        return torch.cat(outputs, dim=1)


class ChronoNet(nn.Module):
    """ChronoNet-style model for spectrogram inputs (2D)."""

    def __init__(self, model_params, metadata):
        super().__init__()

        in_channels = metadata.get("num_channels", 4)
        num_classes = metadata.get("num_classes", 2)

        conv_filters = model_params.get("conv_filters", 32)
        conv_kernel_sizes = tuple(model_params.get("conv_kernel_sizes", [2, 4, 8]))
        conv_stride = model_params.get("conv_stride", 2)
        num_blocks = model_params.get("num_blocks", 3)

        gru_hidden_size = model_params.get("gru_hidden_size", 32)
        dropout_rate = model_params.get("dropout_rate", 0.3)
        self.window_chunk_size = model_params.get("window_chunk_size", 256)
        use_batch_norm = model_params.get("use_batch_norm", True)

        logger.log_dict(model_params)
        logger.log("num_input_channels", in_channels)
        logger.log("num_classes", num_classes)

        blocks = []
        block_in_channels = in_channels
        for _ in range(num_blocks):
            block = ChronoBlock2D(
                block_in_channels,
                conv_filters,
                kernel_sizes=conv_kernel_sizes,
                stride=conv_stride,
            )
            blocks.append(block)
            block_in_channels = conv_filters * len(conv_kernel_sizes)

        self.blocks = nn.ModuleList(blocks)
        self.bn = nn.BatchNorm2d(block_in_channels) if use_batch_norm else None

        self.gru1 = nn.GRU(input_size=block_in_channels, hidden_size=gru_hidden_size, batch_first=True)
        self.gru2 = nn.GRU(input_size=gru_hidden_size, hidden_size=gru_hidden_size, batch_first=True)
        self.gru3 = nn.GRU(input_size=gru_hidden_size * 2, hidden_size=gru_hidden_size, batch_first=True)
        self.gru4 = nn.GRU(input_size=gru_hidden_size * 3, hidden_size=gru_hidden_size, batch_first=True)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(gru_hidden_size, num_classes)

    def encode_window(self, x):
        for block in self.blocks:
            x = block(x)

        if self.bn is not None:
            x = self.bn(x)

        batch_size, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, height * width, channels)

        gru_out1, _ = self.gru1(x)
        gru_out2, _ = self.gru2(gru_out1)
        gru_out12 = torch.cat([gru_out1, gru_out2], dim=2)
        gru_out3, _ = self.gru3(gru_out12)
        gru_out123 = torch.cat([gru_out1, gru_out2, gru_out3], dim=2)
        _, gru_out4 = self.gru4(gru_out123)

        embedding = gru_out4[-1]
        embedding = self.dropout(embedding)
        return embedding

    def classify(self, x):
        return self.fc(x)

    def forward(self, x):
        x = self.encode_window(x)
        return self.classify(x)

    def forward_person(self, windows):
        """
        Aggregate window embeddings for a person using mean pooling.
        windows: Tensor of shape (num_windows, C, H, W) or (B, num_windows, C, H, W)
        """
        device = next(self.parameters()).device
        chunk_size = self.window_chunk_size or 0

        if windows.dim() == 4:
            num_windows = windows.size(0)
            if chunk_size <= 0 or num_windows <= chunk_size:
                if windows.device != device:
                    windows = windows.to(device)
                embeddings = self.encode_window(windows)
                pooled = embeddings.mean(dim=0, keepdim=True)
                return self.classify(pooled)

            embed_sum = None
            for start in range(0, num_windows, chunk_size):
                end = min(start + chunk_size, num_windows)
                window_chunk = windows[start:end]
                if window_chunk.device != device:
                    window_chunk = window_chunk.to(device)
                chunk_embeds = self.encode_window(window_chunk)
                chunk_sum = chunk_embeds.sum(dim=0)
                embed_sum = chunk_sum if embed_sum is None else embed_sum + chunk_sum

            pooled = (embed_sum / num_windows).unsqueeze(0)
            return self.classify(pooled)

        if windows.dim() == 5:
            batch_size, num_windows, channels, height, width = windows.shape
            if chunk_size <= 0 or num_windows <= chunk_size:
                if windows.device != device:
                    windows = windows.to(device)
                windows = windows.view(batch_size * num_windows, channels, height, width)
                embeddings = self.encode_window(windows)
                embeddings = embeddings.view(batch_size, num_windows, -1)
                pooled = embeddings.mean(dim=1)
                return self.classify(pooled)

            embed_sum = None
            for start in range(0, num_windows, chunk_size):
                end = min(start + chunk_size, num_windows)
                window_chunk = windows[:, start:end]
                if window_chunk.device != device:
                    window_chunk = window_chunk.to(device)
                window_chunk = window_chunk.view(batch_size * (end - start), channels, height, width)
                chunk_embeds = self.encode_window(window_chunk)
                chunk_embeds = chunk_embeds.view(batch_size, end - start, -1).sum(dim=1)
                embed_sum = chunk_embeds if embed_sum is None else embed_sum + chunk_embeds

            pooled = embed_sum / num_windows
            return self.classify(pooled)

        raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(windows.shape)}")
