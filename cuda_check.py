import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU设备: {torch.cuda.get_device_name(0)}")
print(f"GPU计算能力: {torch.cuda.get_device_capability(0)}")

# 检查当前PyTorch支持的计算能力
print(f"PyTorch编译时的计算能力: {torch.cuda.get_arch_list()}")