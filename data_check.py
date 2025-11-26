import os, glob
import numpy as np

root_npz = "./data/project_TransUNet/data/Synapse/train_npz"  # 你的 root_path

all_npz = sorted(glob.glob(os.path.join(root_npz, "*.npz")))
print("npz 文件数量:", len(all_npz))

global_unique = set()
min_label = +1e9
max_label = -1e9

for p in all_npz:
    data = np.load(p)
    try:
        label = data["label"]
    except:
        label = data["seg"]  # 以防键名不同

    u = np.unique(label)
    global_unique |= set(u.tolist())
    min_label = min(min_label, u.min())
    max_label = max(max_label, u.max())

print("全数据集 label 最小值:", min_label)
print("全数据集 label 最大值:", max_label)
print("全数据集所有标签值:", sorted(global_unique))
