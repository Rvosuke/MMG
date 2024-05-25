# mg_dataset.py

import os
import csv
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor


class MultiModalGlaucomaDataset(Dataset):
    """
    多模态青光眼数据集的 PyTorch Dataset 类。

    该数据集包含青光眼患者的多模态数据，包括3副眼底图像和多个表格数据。
    该类负责加载、预处理数据，并提供了通过下标访问单个样本的功能。

    Args:
        args: 命令行参数。

    Attributes:
        base_dir (str): 数据集的根目录。
        image_dir (str): 图像数据的目录。
        args: 命令行参数。
        im_ids (List[str]): 图像的ID列表。
        images (List[str]): 图像文件的路径列表。
        info (List[List[str]]): 每个样本的附加信息列表。
        composed_transforms (Compose): 数据预处理和增强的转换流水线。

    """

    def __init__(self, args):
        super(self).__init__()
        self.base_dir = './data'
        self.image_dir = os.path.join(self.base_dir, 'images')
        self.args = args

        self.im_ids, self.images, self.info = self._load_data_info()

        self.composed_transforms = Compose([
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])

    def _load_data_info(self) -> Tuple[List[str], List[str], List[List[str]]]:
        """
        加载数据集的信息。

        Returns:
            一个元组，包含图像ID列表、图像路径列表和附加信息列表。

        """
        with open("./data/info-clinical.csv", "r", errors='ignore') as f:
            info_csv = {row[0]: row[1:] for row in csv.reader(f)}

        with open(os.path.join(self.base_dir, 'val.txt'), "r") as f:
            lines = f.read().splitlines()

        im_ids, images, info = [], [], []
        for line in lines:
            im_ids.append(line)
            images.append(os.path.join(self.image_dir, line))
            info.append(info_csv[line])

        return im_ids, images, info

    def __len__(self) -> int:
        """
        返回数据集中的样本数量。

        Returns:
            数据集中的样本数量。

        """
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """
        通过下标访问单个样本。

        Args:
            index: 要访问的样本的下标。

        Returns:
            一个字典，包含图像数据、标签和样本ID。

        """
        img, target = self._make_img_gt_point_pair(index)
        sample = {'image': img, 'label': target, 'id': self.images[index]}
        return self.composed_transforms(sample)

    def _make_img_gt_point_pair(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建图像-标签对。

        Args:
            index: 要访问的样本的下标。

        Returns:
            一个元组，包含图像数据和标签。

        """
        img = np.concatenate([self._process_image(index, i) for i in [2, 5, 2]], axis=0)
        target = self._create_target(index)
        return img, target

    def _process_image(self, index: int, i: int) -> np.ndarray:
        """
        处理单个图像。

        Args:
            index: 要处理的样本的下标。
            i: 图像的模态索引。

        Returns:
            处理后的图像数据。

        """
        if int(self.info[index][i]):
            pic = Image.open(os.path.join(self.images[index], f'{i + 1}.jpg'))
            pic = pic.resize((512, 512))
        else:
            pic = np.zeros((512, 512, 3))
        return pic[np.newaxis, :]

    def _create_target(self, index: int) -> np.ndarray:
        """
        创建标签。

        Args:
            index: 要处理的样本的下标。

        Returns:
            创建的标签数据。

        """
        target = np.zeros(9)
        target[0] = int(self.info[index][8]) == 0
        target[1] = int(self.info[index][8]) > 0
        target[4:9] = self.info[index][9:14]
        target[5] /= 100
        target[-1] /= 100
        return target
