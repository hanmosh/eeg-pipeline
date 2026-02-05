import torch
from torch import nn
from utils.log import logger


class BelongingCNN(nn.Module):
    """
    Multi-channel CNN for predicting student belongingness from EEG spectrograms.
    Based on architecture from sabotage detection paper, simplified for binary classification.
    """
    
    def __init__(self, model_params, metadata):
        super(BelongingCNN, self).__init__()
        
        num_channels = metadata.get('num_channels', 4)
        image_size = metadata.get('image_size', (64, 64))
        num_classes = metadata.get('num_classes', 2)
        
        dropout_rate = model_params.get('dropout_rate', 0.3)
        conv1_filters = model_params.get('conv1_filters', 32)
        conv2_filters = model_params.get('conv2_filters', 64)
        fc_hidden_size = model_params.get('fc_hidden_size', 128)
        self.window_chunk_size = model_params.get('window_chunk_size', 256)
        
        logger.log_dict(model_params)
        logger.log('num_input_channels', num_channels)
        logger.log('image_size', str(image_size))
        logger.log('num_classes', num_classes)
        
        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=conv1_filters,
            kernel_size=5,
            padding=2
        )
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(
            in_channels=conv1_filters,
            out_channels=conv2_filters,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(conv2_filters)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        conv_output_size = conv2_filters * (image_size[0] // 4) * (image_size[1] // 4)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(conv_output_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, num_classes)
        
        self.relu = nn.ReLU()
        
    def encode_window(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        
        return x

    def classify(self, x):
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.encode_window(x)
        x = self.classify(x)
        return x

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
