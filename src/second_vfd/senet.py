import torch
import torch.nn as nn
from typing import Tuple, List


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""

    def __init__(self, in_channels: int, filters: Tuple[int, int, int], stride: int = 1, down_sample: nn.Module = None):
        """
        Args:
            in_channels (int): 输入通道数
            filters (Tuple[int, int, int]): 三个卷积层的输出通道数
            stride (int): 第一个卷积层的步幅,默认为1
            down_sample (nn.Module): 下采样层,用于调整输入的维度,默认为None
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, filters[0], kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(filters[2], filters[2] // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[2] // 16, filters[2], kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量,形状为(batch_size, in_channels, height, width)

        Returns:
            torch.Tensor: 输出张量,形状为(batch_size, filters[2], height, width)
        """
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        se = self.se(out)
        out = out * se

        if self.down_sample is not None:
            identity = self.down_sample(x)
        out += identity
        out = self.relu(out)
        return out


def make_layer(in_channels: int, filters: Tuple[int, int, int], num_blocks: int, stride: int = 1) -> nn.Sequential:
    """创建由多个SEBlock组成的层

    Args:
        in_channels (int): 输入通道数
        filters (Tuple[int, int, int]): 三个卷积层的输出通道数
        num_blocks (int): SEBlock的数量
        stride (int): 第一个SEBlock的步幅,默认为1

    Returns:
        nn.Sequential: 由多个SEBlock组成的层
    """
    down_sample = None
    if stride != 1 or in_channels != filters[2]:
        down_sample = nn.Sequential(
            nn.Conv2d(in_channels, filters[2], kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(filters[2])
        )

    layers = [SEBlock(in_channels, filters, stride, down_sample)]
    for _ in range(1, num_blocks):
        layers.append(SEBlock(filters[2], filters))
    return nn.Sequential(*layers)


class SENet(nn.Module):
    """Squeeze-and-Excitation Network"""

    def __init__(self, num_classes: int, block_config: List[int]):
        """
        Args:
            num_classes (int): 类别数
            block_config (List[int]): 每个层包含的SEBlock数量
        """
        super().__init__()
        self.conv3 = make_layer(768, (1024, 1024, 2048), block_config[1], stride=2)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入张量,形状为(batch_size, 768, height, width)

        Returns:
            torch.Tensor: 输出张量,形状为(batch_size, num_classes)
        """
        x = self.conv3(x)
        x = x.mean(dim=[2, 3])  # Global average pooling
        x = self.fc(x)
        return x


def attention_cnn(num_classes: int = 2) -> SENet:
    """创建SENet模型

    Args:
        num_classes (int): 类别数,默认为2

    Returns:
        SENet: SENet模型实例
    """
    block_config = [3, 4, 6, 3]
    return SENet(num_classes, block_config)


if __name__ == "__main__":
    # net = attention_cnn()
    net = SENet(2, [3, 4, 6, 3])
    x = torch.randn(10, 768, 14, 14)
    y = net(x)
    print(y.size())
