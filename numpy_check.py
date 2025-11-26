import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp

print("=== 环境检查 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"NumPy版本: {np.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# 测试简单的数据集
class TestDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # 测试numpy在worker中是否可用
        try:
            data = np.random.randn(3, 224, 224).astype(np.float32)
            data = torch.from_numpy(data)
            return data, idx
        except Exception as e:
            print(f"Error in worker: {e}")
            raise

# 测试不同worker数量
for num_workers in [0, 1, 2]:
    print(f"\n=== 测试 num_workers={num_workers} ===")
    try:
        dataset = TestDataset()
        dataloader = DataLoader(dataset, batch_size=4, num_workers=num_workers)
        
        for i, (data, idx) in enumerate(dataloader):
            if i >= 2:  # 只测试前两个batch
                break
            print(f"Batch {i}: {data.shape}")
        print(f"num_workers={num_workers}: 成功")
    except Exception as e:
        print(f"num_workers={num_workers}: 失败 - {e}")