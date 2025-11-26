import os
import random

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

class MedicalNoiseInjector2D:
    """
    2D医疗图像分割标签噪声注入器
    专为Synapse等2D切片数据集设计
    """
    
    def __init__(self, num_classes: int, seed: int = 42):
        # 步骤1.1: 初始化参数
        self.num_classes = num_classes
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        
    def random_label_noise(self, labels: np.ndarray, noise_ratio: float = 0.1):
        """步骤1.2: 实现随机标签噪声注入"""
        # 创建标签的副本，避免修改原始数据
        noisy_labels = labels.copy()
        h, w = labels.shape
        
        # 步骤1.3: 创建随机掩码来选择要注入噪声的像素
        mask = np.random.random(labels.shape) < noise_ratio
        
        # 步骤1.4: 排除背景类别 (class 0)
        mask[labels == 0] = False
        
        # 步骤1.5: 为每个类别处理噪声注入
        for class_id in range(1, self.num_classes):
            # 找到当前类别且被选中的像素
            class_mask = mask & (labels == class_id)
            if np.any(class_mask):
                # 随机选择新标签（不包括原标签）
                possible_new_classes = [c for c in range(1, self.num_classes) if c != class_id]
                if possible_new_classes:
                    new_classes = np.random.choice(
                        possible_new_classes, 
                        size=np.sum(class_mask)
                    )
                    noisy_labels[class_mask] = new_classes
        # print(f"Injected random noise with ratio {noise_ratio}")

        return noisy_labels
    
    def boundary_aware_noise(self, labels: np.ndarray, noise_ratio: float = 0.1, boundary_width: int = 2):
        """步骤2.1: 实现边界感知噪声注入"""
        noisy_labels = labels.copy()
        
        # 步骤2.2: 为每个器官类别计算边界
        for class_id in range(1, self.num_classes):
            # 创建二值掩码
            binary_mask = (labels == class_id).astype(np.uint8)
            
            if np.sum(binary_mask) == 0:
                continue
                
            # 步骤2.3: 计算距离变换获取边界区域
            distance = ndimage.distance_transform_edt(binary_mask)
            boundary_mask = (distance > 0) & (distance <= boundary_width)
            
            if not np.any(boundary_mask):
                continue
                
            # 步骤2.4: 边界区域使用更高噪声比例
            boundary_noise_ratio = noise_ratio * 2.0
            
            # 步骤2.5: 在边界区域注入噪声
            boundary_change_mask = boundary_mask & (np.random.random(labels.shape) < boundary_noise_ratio)
            
            # 随机选择相邻类别作为新标签
            possible_classes = [c for c in range(1, self.num_classes) if c != class_id]
            if possible_classes:
                new_classes = np.random.choice(
                    possible_classes, 
                    size=np.sum(boundary_change_mask)
                )
                noisy_labels[boundary_change_mask] = new_classes
        # print(f"Injected boundary-aware noise with ratio {noise_ratio} and boundary width {boundary_width}")

        return noisy_labels

    def simulate_expert_variability(self, labels: np.ndarray, confusion_pairs: list, noise_ratio: float = 0.1):
        """步骤2.6: 实现专家变异性噪声"""
        noisy_labels = labels.copy()
        
        # 步骤2.7: 处理每个容易混淆的器官对
        for organ_a, organ_b in confusion_pairs:
            # 在两个方向上都可能混淆
            for src, dst in [(organ_a, organ_b), (organ_b, organ_a)]:
                src_mask = labels == src
                if not np.any(src_mask):
                    continue
                    
                # 选择要改变的部分
                change_mask = src_mask & (np.random.random(labels.shape) < noise_ratio)
                noisy_labels[change_mask] = dst
        # print(f"Injected expert variability noise with ratio {noise_ratio} for confusion pairs {confusion_pairs}")

        return noisy_labels

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, noise_injector=None, noise_config=None):
        self.output_size = output_size
        self.noise_injector = noise_injector
        self.noise_config = noise_config
        print(f"RandomGenerator initialized with noise_injector: {noise_injector is not None}, noise_config: {noise_config}")

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        if self.noise_injector and self.noise_config:
            noise_type = self.noise_config.get('type', 'random')
            noise_ratio = self.noise_config.get('ratio', 0.1)
            # print(f"Attempting to inject {noise_type} noise with ratio {noise_ratio}")
            
            # 步骤3.5: 根据配置选择噪声类型
            if noise_type == 'random':
                label = self.noise_injector.random_label_noise(label, noise_ratio)
            elif noise_type == 'boundary':
                boundary_width = self.noise_config.get('boundary_width', 2)
                label = self.noise_injector.boundary_aware_noise(label, noise_ratio, boundary_width)
            elif noise_type == 'expert':
                confusion_pairs = self.noise_config.get('confusion_pairs', [(1, 2)])
                label = self.noise_injector.simulate_expert_variability(label, confusion_pairs, noise_ratio)
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, noise_config=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir
        self.noise_config = noise_config

        # 步骤4.2: 仅在训练集上初始化噪声注入器
        if self.split == "train" and noise_config is not None:
            self.noise_injector = MedicalNoiseInjector2D(num_classes=9)
            print(f"Noise injector initialized for {split} with config: {noise_config}")
        else:
            self.noise_injector = None
            print(f"No noise injector for {split}")

        if self.noise_injector is not None and self.transform is not None:
            self._setup_noise_injection()

    def _setup_noise_injection(self):
        """步骤5: 设置噪声注入，处理Compose包装的情况"""
        if hasattr(self.transform, 'transforms'):
            # transform是Compose对象，遍历其中的所有transform
            for i, t in enumerate(self.transform.transforms):
                if isinstance(t, RandomGenerator):
                    # 找到RandomGenerator，设置噪声注入器
                    self.transform.transforms[i].noise_injector = self.noise_injector
                    self.transform.transforms[i].noise_config = self.noise_config
                    print(f"Set noise injector for RandomGenerator at index {i}")
                    break
        elif isinstance(self.transform, RandomGenerator):
            # transform直接是RandomGenerator
            self.transform.noise_injector = self.noise_injector
            self.transform.noise_config = self.noise_config
            print("Set noise injector for direct RandomGenerator")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split in ["train", "val"] or self.sample_list[idx].strip('\n').split(",")[0].endswith(".npz"):
            slice_name = self.sample_list[idx].strip('\n').split(",")[0]
            if slice_name.endswith(".npz"):
                data_path = os.path.join(self.data_dir, slice_name)
            else:
                data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            try:
                image, label = data['image'], data['label']
            except:
                image, label = data['data'], data['seg']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        # if self.transform:
        #     # sample = self.transform(sample)
        #     # 如果transform是RandomGenerator且配置了噪声，传递噪声注入器
        #     if isinstance(self.transform, RandomGenerator) and self.noise_injector is not None:
        #         # 步骤4.5: 为RandomGenerator设置噪声注入器
        #         self.transform.noise_injector = self.noise_injector
        #         sample = self.transform(sample)
        #     else:
        #         sample = self.transform(sample)
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
