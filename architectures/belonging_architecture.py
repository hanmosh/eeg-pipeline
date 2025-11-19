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
        
    def forward(self, x):
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
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x