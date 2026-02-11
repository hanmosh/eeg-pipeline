import math
import torch
from torch import nn
import torch.nn.functional as F


class Conv2dSame(nn.Module):
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
        return self.conv(self._pad_same(x))


class ChronoBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(2, 4, 8), stride=2):
        super().__init__()
        self.convs = nn.ModuleList([
            Conv2dSame(in_channels, out_channels, kernel_size=ks, stride=stride)
            for ks in kernel_sizes
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        outputs = [self.relu(conv(x)) for conv in self.convs]
        return torch.cat(outputs, dim=1)


class ChronoNet(nn.Module):
    """ChronoNet for window sequences with temporal GRU aggregation."""

    def __init__(self, model_params, metadata):
        super().__init__()

        in_channels = metadata.get('num_channels', 4)
        num_classes = metadata.get('num_classes', 2)

        conv_filters = model_params.get('conv_filters', 32)
        conv_kernel_sizes = tuple(model_params.get('conv_kernel_sizes', [2, 4, 8]))
        conv_stride = model_params.get('conv_stride', 2)
        num_blocks = model_params.get('num_blocks', 3)

        gru_hidden_size = model_params.get('gru_hidden_size', 32)
        dropout_rate = model_params.get('dropout_rate', 0.3)
        use_batch_norm = model_params.get('use_batch_norm', True)

        temporal_hidden_size = model_params.get('temporal_gru_hidden_size', gru_hidden_size)
        temporal_num_layers = model_params.get('temporal_num_layers', 1)
        temporal_bidirectional = model_params.get('temporal_bidirectional', False)
        temporal_dropout_rate = model_params.get('temporal_dropout_rate', dropout_rate)
        self.window_batch_size = model_params.get('window_batch_size', None)
        self.truncate_bptt = model_params.get('truncate_bptt', None)

        blocks = []
        block_in_channels = in_channels
        for _ in range(num_blocks):
            blocks.append(
                ChronoBlock2D(
                    block_in_channels,
                    conv_filters,
                    kernel_sizes=conv_kernel_sizes,
                    stride=conv_stride,
                )
            )
            block_in_channels = conv_filters * len(conv_kernel_sizes)

        self.blocks = nn.ModuleList(blocks)
        self.bn = nn.BatchNorm2d(block_in_channels) if use_batch_norm else None

        self.gru1 = nn.GRU(input_size=block_in_channels, hidden_size=gru_hidden_size, batch_first=True)
        self.gru2 = nn.GRU(input_size=gru_hidden_size, hidden_size=gru_hidden_size, batch_first=True)
        self.gru3 = nn.GRU(input_size=gru_hidden_size * 2, hidden_size=gru_hidden_size, batch_first=True)
        self.gru4 = nn.GRU(input_size=gru_hidden_size * 3, hidden_size=gru_hidden_size, batch_first=True)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(gru_hidden_size, num_classes)

        self.temporal_gru = nn.GRU(
            input_size=gru_hidden_size,
            hidden_size=temporal_hidden_size,
            num_layers=temporal_num_layers,
            batch_first=True,
            bidirectional=temporal_bidirectional,
            dropout=temporal_dropout_rate if temporal_num_layers > 1 else 0.0,
        )
        temporal_out_size = temporal_hidden_size * (2 if temporal_bidirectional else 1)
        self.temporal_dropout = nn.Dropout(temporal_dropout_rate)
        self.temporal_fc = nn.Linear(temporal_out_size, num_classes)

    def encode_window(self, x):
        for block in self.blocks:
            x = block(x)
        if self.bn is not None:
            x = self.bn(x)

        batch_size, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)

        gru_out1, _ = self.gru1(x)
        gru_out2, _ = self.gru2(gru_out1)
        gru_out12 = torch.cat([gru_out1, gru_out2], dim=2)
        gru_out3, _ = self.gru3(gru_out12)
        gru_out123 = torch.cat([gru_out1, gru_out2, gru_out3], dim=2)
        _, gru_out4 = self.gru4(gru_out123)

        embedding = self.dropout(gru_out4[-1])
        return embedding

    def forward_sequence(self, windows, lengths=None):
        batch_size, num_windows, channels, height, width = windows.shape
        if self.truncate_bptt and self.truncate_bptt > 0:
            if self.temporal_gru.bidirectional:
                raise ValueError("truncate_bptt is not supported with bidirectional temporal GRU.")

            trunc = int(self.truncate_bptt)
            lengths_cpu = lengths.cpu() if lengths is not None else torch.full(
                (batch_size,), num_windows, dtype=torch.long
            )
            hidden = None
            last_out = torch.zeros(
                (batch_size, self.temporal_gru.hidden_size),
                device=windows.device,
                dtype=windows.dtype,
            )

            for start in range(0, num_windows, trunc):
                end = min(start + trunc, num_windows)
                chunk = windows[:, start:end]
                chunk_len = end - start

                chunk_flat = chunk.view(batch_size * chunk_len, channels, height, width)
                total_chunk = chunk_flat.size(0)
                if self.window_batch_size and self.window_batch_size > 0 and total_chunk > self.window_batch_size:
                    embeddings_chunks = []
                    for w_start in range(0, total_chunk, self.window_batch_size):
                        w_end = min(w_start + self.window_batch_size, total_chunk)
                        embeddings_chunks.append(self.encode_window(chunk_flat[w_start:w_end]))
                    embeddings = torch.cat(embeddings_chunks, dim=0)
                else:
                    embeddings = self.encode_window(chunk_flat)
                embeddings = embeddings.view(batch_size, chunk_len, -1)

                lengths_chunk = (lengths_cpu - start).clamp(min=0, max=chunk_len)
                active_mask = lengths_chunk > 0
                if not torch.any(active_mask):
                    break
                active_idx = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
                embeddings_active = embeddings[active_idx]
                lengths_active = lengths_chunk[active_idx]

                packed = nn.utils.rnn.pack_padded_sequence(
                    embeddings_active, lengths_active, batch_first=True, enforce_sorted=False
                )
                hidden_active = hidden[:, active_idx, :] if hidden is not None else None
                _packed_out, hidden_new = self.temporal_gru(packed, hidden_active)

                if hidden is None:
                    hidden = torch.zeros(
                        self.temporal_gru.num_layers,
                        batch_size,
                        self.temporal_gru.hidden_size,
                        device=windows.device,
                        dtype=embeddings.dtype,
                    )
                hidden[:, active_idx, :] = hidden_new
                last_out[active_idx] = hidden_new[-1]
                hidden = hidden.detach()

            last_out = self.temporal_dropout(last_out)
            return self.temporal_fc(last_out)

        windows = windows.view(batch_size * num_windows, channels, height, width)
        total_windows = windows.size(0)
        if self.window_batch_size and self.window_batch_size > 0 and total_windows > self.window_batch_size:
            embeddings_chunks = []
            for start in range(0, total_windows, self.window_batch_size):
                end = min(start + self.window_batch_size, total_windows)
                embeddings_chunks.append(self.encode_window(windows[start:end]))
            embeddings = torch.cat(embeddings_chunks, dim=0)
        else:
            embeddings = self.encode_window(windows)
        embeddings = embeddings.view(batch_size, num_windows, -1)

        if lengths is not None:
            lengths_cpu = lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embeddings, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.temporal_gru(packed)
            temporal_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=num_windows
            )
            idx = (lengths_cpu - 1).clamp(min=0).long().to(temporal_out.device)
            batch_idx = torch.arange(batch_size, device=temporal_out.device)
            last_out = temporal_out[batch_idx, idx]
        else:
            temporal_out, _ = self.temporal_gru(embeddings)
            last_out = temporal_out[:, -1, :]

        last_out = self.temporal_dropout(last_out)
        return self.temporal_fc(last_out)

    def forward(self, x, lengths=None):
        if x.dim() == 5:
            return self.forward_sequence(x, lengths=lengths)
        return self.fc(self.encode_window(x))
