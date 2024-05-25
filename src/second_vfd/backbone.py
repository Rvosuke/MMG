"""
SeNet网络结构和实现

1. 网络结构
   - 主干网络: 由多个卷积层和池化层组成,用于提取输入图像的特征。
   - SE注意力机制: 在每个残差块中引入SE注意力机制,自适应地调整通道之间的重要性。
   - 全局平均池化: 在主干网络的末尾使用全局平均池化,将特征图缩减为固定大小。
   - 全连接层: 将池化后的特征传递给全连接层,生成最终的类别预测。

2. 实现细节
   - Block类: 实现了一个残差块,包含两个卷积层和一个SE注意力模块。
   - make_layer函数: 用于创建由多个残差块组成的层。
   - SeNet类: 实现了完整的SeNet网络结构。

3. 设计思想
   - 残差连接: 在残差块中使用恒等映射,缓解梯度消失问题,加深网络深度。
   - SE注意力机制: 通过自适应地调整通道之间的重要性,提高特征表示能力。
   - 全局平均池化: 将特征图缩减为固定大小,减少参数数量,提高计算效率。

4. 输入输出
   - 输入: (batch_size, 3, height, width)
   - 输出: (batch_size, classes)
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class Block(nn.Module):
    """实现残差块,包含两个卷积层和一个SE注意力模块"""

    def __init__(self, in_channels: int, filters: Tuple[int, int, int], stride: int = 1, use_1x1conv: bool = False):
        """
        Args:
            in_channels: 输入通道数
            filters: 卷积层的输出通道数列表
            stride: 卷积层的步长
            use_1x1conv: 是否使用1x1卷积层进行下采样
        """
        super(Block, self).__init__()
        filter1, filter2, filter3 = filters
        self.is_1x1conv = use_1x1conv
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, filter1, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(filter1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(filter1, filter2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filter2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(filter2, filter3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(filter3),
        )

        if use_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filter3, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filter3)
            )
        else:
            self.shortcut = nn.Identity()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(filter3, filter3 // 16, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter3 // 16, filter3, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        se_weights = self.se(x)
        x = x * se_weights
        x += shortcut
        return nn.ReLU(inplace=True)(x)


class SENet(nn.Module):
    """实现SeNet网络结构"""
    def __init__(self, num_classes: int, block_nums: List[int]):
        """
        Args:
            num_classes: 类别数
            block_nums: 每个层中残差块的个数
        """
        super(SENet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self.make_layer(64, (64, 64, 256), block_nums[0])
        self.conv3 = self.make_layer(256, (128, 128, 512), block_nums[1], stride=2)
        self.conv4 = self.make_layer(512, (256, 256, 1024), block_nums[2], stride=2)
        self.conv5 = self.make_layer(1024, (512, 512, 2048), block_nums[3], stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量
        Returns:
            网络的输出
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @staticmethod
    def make_layer(in_channels: int, filters: Tuple[int, int, int], num_blocks: int, stride: int = 1) -> nn.Sequential:
        """
        创建由多个残差块组成的层
        Args:
            in_channels: 输入通道数
            filters: 卷积层的输出通道数列表
            num_blocks: 残差块的个数
            stride: 第一个残差块的步长
        Returns:
            由多个残差块组成的Sequential
        """
        layers = [Block(in_channels, filters, stride=stride, use_1x1conv=True)]
        for _ in range(1, num_blocks):
            layers.append(Block(filters[2], filters))
        return nn.Sequential(*layers)


if __name__ == "__main__":
    net = SENet(2, [3, 4, 6, 3])
    inputs = torch.randn(10, 3, 224, 224)
    outputs = net(inputs)
    print(outputs.size())
