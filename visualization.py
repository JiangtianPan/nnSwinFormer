import os
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

pred_dir = Path("./test_outputs_synapse/predictions")
image_dir = Path("./data/project_TransUNet/data/Synapse/images")
out_dir = Path("./viz_png")
out_dir.mkdir(parents=True, exist_ok=True)

def center_slices(mask3d, axis=2, k=3):
    """从有前景的区域里挑 k 个代表切片（中位数及两侧）"""
    fg = (mask3d > 0).any(axis=(axis^1, axis^2)) if mask3d.ndim==3 else None
    # 简便：直接用中间+两侧
    n = mask3d.shape[axis]
    idxs = [n//2 - n//6, n//2, n//2 + n//6]
    idxs = [int(np.clip(i,0,n-1)) for i in idxs]
    return sorted(set(idxs))[:k]

for pred_path in sorted(pred_dir.glob("*.nii*")):
    case_id = pred_path.stem.replace("_pred","")
    cand_img = [image_dir/f"{case_id}.nii.gz", image_dir/f"{case_id}.nii"]
    img_path = next((p for p in cand_img if p.exists()), None)
    if img_path is None:
        print(f"[skip] {case_id} 找不到原图")
        continue

    img = nib.as_closest_canonical(nib.load(str(img_path))).get_fdata(dtype=np.float32)
    pred = nib.as_closest_canonical(nib.load(str(pred_path))).get_fdata()
    pred = np.rint(pred).astype(np.int16)

    if pred.shape != img.shape:
        from scipy.ndimage import zoom
        pred = zoom(pred, np.array(img.shape)/np.array(pred.shape), order=0).astype(np.int16)

    cmap = build_cmap(num_classes=int(pred.max()))
    for k in center_slices(pred, axis=2, k=3):
        ct = window_ct(img[:,:,k])
        sg = pred[:,:,k]
        plt.figure(figsize=(6,6))
        plt.imshow(ct, cmap='gray'); plt.imshow(sg, cmap=cmap)
        plt.axis('off'); plt.title(f"{case_id} - axial {k}")
        plt.savefig(out_dir/f"{case_id}_axial_{k:03d}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

print(f"导出完成，查看：{out_dir.resolve()}")
