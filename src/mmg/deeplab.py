"""
DeepLab网络结构和处理逻辑

1. 网络结构
   - 主干网络: 使用三个SeNet作为主干网络,用于提取输入图像的特征。
   - 注意力卷积层: 在主干网络之后使用attention_cnn层,用于捕获通道之间的依赖关系。
   - 图像特征处理: 使用平均池化和全连接层处理提取到的图像特征。
   - 临床数据处理: 使用两个全连接层处理输入的临床数据。
   - 特征融合: 将处理后的图像特征和临床数据特征在通道维度上连接。
   - 输出层: 使用两个全连接层和Softmax激活函数生成最终的预测结果。

2. 处理逻辑
   - 输入: 包含三个通道的图像数据和临床数据。
   - 图像特征提取: 将输入图像的三个通道分别传递给三个SeNet主干网络,提取图像特征。
   - 注意力卷积: 将提取到的图像特征传递给attention_cnn层,捕获通道之间的依赖关系。
   - 图像特征处理: 对attention_cnn层的输出进行平均池化,并使用全连接层进行特征变换。
   - 临床数据处理: 将输入的临床数据传递给两个全连接层,提取临床特征。
   - 特征融合: 将处理后的图像特征和临床特征在通道维度上连接。
   - 输出预测: 将融合后的特征传递给两个全连接层,并应用Softmax激活函数生成最终的预测结果。

3. 设计思想
   - 多模态融合: 同时利用图像数据和临床数据,充分利用不同模态的信息。
   - 注意力机制: 通过attention_cnn层捕获通道之间的依赖关系,提高特征表示能力。
   - 残差连接: 在ResidualBlock中使用残差连接,缓解梯度消失问题,加深网络深度。
   - 模块化设计: 将网络划分为多个子模块,如ResidualBlock和make_layer,提高代码的可读性和可维护性。

4. 代码结构
   - ResidualBlock类: 实现残差块,包含左侧分支和可选的右侧shortcut连接。
   - DeepLab类: 实现整个深度学习模型,包括主干网络、注意力卷积层、全连接层等。
   - make_layer函数: 用于创建包含多个残差块的layer层。

5. 输入输出
   - 输入: input_ (batch_size, 3, height, width), clinical (batch_size, 4)
   - 输出: output (batch_size, 2)

6. 优化与改进
   - 可以尝试使用其他主干网络,如ResNet、Inception等,以提高特征提取能力。
   - 可以探索不同的注意力机制,如se_block、cbam等,以进一步增强特征表示能力。
   - 可以引入更多的临床数据或其他模态数据,以充分利用多模态信息。
   - 可以使用更大的数据集和更深的网络结构,以提高模型的性能和泛化能力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.mmg.backbone import SENet
from src.mmg.attention_cnn_layer import attention_cnn


class ResidualBlock(nn.Module):
    """实现子模块：ResidualBlock"""

    def __init__(self, in_channel: int, out_channel: int, stride: int = 1, shortcut: Optional[nn.Module] = None):
        """
        初始化ResidualBlock

        Args:
            in_channel: 输入通道数
            out_channel: 输出通道数
            stride: 卷积步长，默认为1
            shortcut: 右侧的shortcut连接，默认为None
        """
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.right = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ResidualBlock的前向传播函数

        Args:
            x: 输入张量

        Returns:
            ResidualBlock的输出
        """
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class DeepLab(nn.Module):
    def __init__(self, freeze_bn: bool = False):
        """
        初始化DeepLab

        Args:
            freeze_bn: 是否冻结BatchNorm层，默认为False
        """
        super(DeepLab, self).__init__()
        self.backbone = nn.ModuleList([SENet(2, [3, 4, 6, 3]) for _ in range(3)])
        self.freeze_bn = freeze_bn
        self.attention = attention_cnn()
        self.image_fc = nn.Linear(8192, 512)
        self.clinical_fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.Linear(128, 64)
        )
        self.output_fc = nn.Sequential(
            nn.Linear(576, 256),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, input_: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        """
        DeepLab的前向传播函数

        Args:
            input_: 输入图像张量，形状为(batch_size, 3, height, width)
            clinical: 临床数据张量，形状为(batch_size, 4)

        Returns:
            DeepLab的输出，形状为(batch_size, 2)
        """
        features = [net(input_[:, i]) for i, net in enumerate(self.backbone)]
        image_feature = torch.cat(features, dim=1)
        image_feature = self.attention(image_feature)
        image_feature = F.avg_pool2d(image_feature, 7).reshape(image_feature.size(0), -1)
        image_feature = self.image_fc(image_feature)

        clinical_feature = self.clinical_fc(clinical)

        combined_feature = torch.cat((image_feature, clinical_feature), dim=1)
        output = self.output_fc(combined_feature)
        return output

    @staticmethod
    def make_layer(in_channel: int, out_channel: int, block_num: int, stride: int = 1) -> nn.Sequential:
        """
        创建layer层，包含多个ResidualBlock

        Args:
            in_channel: 输入通道数
            out_channel: 输出通道数
            block_num: ResidualBlock的个数
            stride: 第一个ResidualBlock的卷积步长，默认为1

        Returns:
            包含多个ResidualBlock的Sequential
        """
        shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        layers = [ResidualBlock(in_channel, out_channel, stride, shortcut)]
        layers.extend([ResidualBlock(out_channel, out_channel) for _ in range(1, block_num)])
        return nn.Sequential(*layers)
