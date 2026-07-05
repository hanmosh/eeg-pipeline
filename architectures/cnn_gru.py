import torch
from torch import nn


class CNNGRU(nn.Module):
    def __init__(self, model_params, metadata):
        super().__init__()

        in_channels = metadata.get("num_channels", 4)
        num_classes = metadata.get("num_classes", 2)

        conv_filters = model_params.get("conv_filters", 16)
        num_blocks = model_params.get("num_blocks", 2)
        gru_hidden_size = model_params.get("gru_hidden_size", 32)
        dropout_rate = model_params.get("dropout_rate", 0.3)
        use_batch_norm = model_params.get("use_batch_norm", True)
        temporal_num_layers = model_params.get("temporal_num_layers", 1)
        temporal_bidirectional = model_params.get("temporal_bidirectional", False)
        self.window_batch_size = model_params.get("window_batch_size", None)

        layers = []
        block_in_channels = in_channels
        block_out_channels = conv_filters
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(block_in_channels, block_out_channels, kernel_size=3, padding=1))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(block_out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            block_in_channels = block_out_channels
            block_out_channels *= 2

        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(block_in_channels, gru_hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.temporal_gru = nn.GRU(
            input_size=gru_hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=temporal_num_layers,
            batch_first=True,
            bidirectional=temporal_bidirectional,
            dropout=dropout_rate if temporal_num_layers > 1 else 0.0,
        )
        temporal_out_size = gru_hidden_size * (2 if temporal_bidirectional else 1)
        self.fc = nn.Linear(temporal_out_size, num_classes)

    def encode_window(self, x):
        x = self.encoder(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        return self.dropout(x)

    def forward_sequence(self, windows, lengths=None):
        batch_size, num_windows, channels, height, width = windows.shape
        windows = windows.view(batch_size * num_windows, channels, height, width)
        total_windows = windows.size(0)

        if self.window_batch_size and self.window_batch_size > 0 and total_windows > self.window_batch_size:
            chunks = []
            for start in range(0, total_windows, self.window_batch_size):
                end = min(start + self.window_batch_size, total_windows)
                chunks.append(self.encode_window(windows[start:end]))
            embeddings = torch.cat(chunks, dim=0)
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

        return self.fc(self.dropout(last_out))

    def forward(self, x, lengths=None):
        if x.dim() == 5:
            return self.forward_sequence(x, lengths=lengths)
        return self.fc(self.encode_window(x))
