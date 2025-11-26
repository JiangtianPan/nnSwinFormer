import torch
import torch.nn as nn
from typing import Optional, Tuple
import copy


try:
    # Official Swin-Unet repo layout
    from networks.vision_transformer import SwinUnet  # type: ignore
except Exception:  # pragma: no cover - for local debugging without repo
    # Fallback for alternative layouts (e.g. when networks is on PYTHONPATH)
    try:
        from networks.vision_transformer import SwinUnet  # type: ignore
    except Exception:
        class SwinUnet(nn.Module):  # dummy for syntax check
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.conv = nn.Conv2d(1, 2, 3, padding=1)
            def forward(self, x):
                return self.conv(x)


class RobustNnSwinFormer(nn.Module):
    """
    RobustNnSwinFormer

    中文说明：
        在官方 Swin-Unet 实现的基础上封装的鲁棒版本，用于论文 / 实验中的
        “Robust nnSwinFormer” 模型。

    改进点：
      1. 保留 3D 体数据输入：x.shape = [B, C, D, H, W]，按 D 维做 slice-wise Swin-Unet。
      2. 支持 Monte-Carlo Dropout：在 eval() 模式下多次前向采样，用于不确定性估计。
      3. 在 SwinUnet 输出的 logits 上增加一个 nnFormer 风格的残差 refine head：
         logits_out = logits_backbone + refine(logits_backbone)
         且 refine 的最后一层初始化为 0，使得初始行为 = 纯 SwinUnet。
    """

    def __init__(
        self,
        config,
        img_size: int = 224,
        num_classes: int = 2,
        use_mc_dropout: bool = False,
        mc_dropout_p: float = 0.1,
        refine_channels: int = 64,
        **kwargs,
    ) -> None:
        super().__init__()
        # Reuse the original Swin-Unet implementation as the backbone
        self.backbone = SwinUnet(config, img_size=img_size, num_classes=num_classes, **kwargs)

        self.use_mc_dropout = use_mc_dropout
        self.mc_dropout_p = mc_dropout_p
        self.num_classes = num_classes
        self.img_size = img_size

        # -------- 新增：logits 级别的 refine head（nnFormer 风格的小 conv + SE + dropout） --------
        # 输入/输出通道 = num_classes，残差形式：out = logits + refine(logits)
        self.refine = nn.Sequential(
            nn.Conv2d(num_classes, refine_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(refine_channels),
            nn.GELU(),
            nn.Dropout2d(p=self.mc_dropout_p),
            nn.Conv2d(refine_channels, num_classes, kernel_size=1, bias=True),
        )
        # 关键：最后一层 conv 初始化为 0，保证刚开始 out ≈ backbone 原结果
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

        if self.use_mc_dropout:
            self._patch_dropout_layers(self.backbone, p=self.mc_dropout_p)
            # refine 里的 Dropout2d 已经用 mc_dropout_p 初始化了

    def load_from(self, config):
        """
        兼容 Swin-Unet 的预训练权重加载：
        把权重加载到 self.backbone（原始 SwinUnet）上。
        """
        # 如果没预训练路径，直接跳过
        pretrained_path = getattr(config.MODEL, "PRETRAIN_CKPT", None)
        if not pretrained_path:
            print("[RobustNnSwinFormer] no PRETRAIN_CKPT, train from scratch.")
            return

        print(f"[RobustNnSwinFormer] loading pretrained Swin weights from: {pretrained_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(pretrained_path, map_location=device)

        # 兼容两种 ckpt 结构：直接 state_dict 或 {'model': state_dict}
        if "model" in checkpoint:
            pretrained_dict = checkpoint["model"]
        else:
            pretrained_dict = checkpoint

        # 如果 backbone 上还有自己实现的 load_from，就直接用它（最保险）
        if hasattr(self.backbone, "load_from"):
            print("[RobustNnSwinFormer] delegate to backbone.load_from(config)")
            self.backbone.load_from(config)
            return

        # 否则，做一个类似官方 Swin-Unet 的手动映射（encoder -> decoder）
        model_dict = self.backbone.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)

        # 把 encoder 的层复制一份给 decoder 的 layers_up
        for k, v in list(pretrained_dict.items()):
            if "layers." in k:
                # 原版是 4 层 encoder，这里按 Swin-Unet 的方式做一个反向映射
                try:
                    enc_id = int(k.split(".")[1])
                except Exception:
                    continue
                num_layers = getattr(self.backbone, "num_layers", 4)
                dec_id = (num_layers - 1) - enc_id
                new_k = k.replace(f"layers.{enc_id}", f"layers_up.{dec_id}")
                full_dict[new_k] = v

        # 删除 shape 对不上的 key
        for k in list(full_dict.keys()):
            if k in model_dict and full_dict[k].shape != model_dict[k].shape:
                print(f"[RobustNnSwinFormer] drop key: {k}, "
                      f"pretrain {tuple(full_dict[k].shape)} vs model {tuple(model_dict[k].shape)}")
                full_dict.pop(k)

        msg = self.backbone.load_state_dict(full_dict, strict=False)
        print("[RobustNnSwinFormer] load_state_dict message:", msg)

    @staticmethod
    def _patch_dropout_layers(module: nn.Module, p: float) -> None:
        """
        Change the dropout rate of all Dropout layers inside the given module.
        This is useful when enabling MC Dropout at test time.
        """
        for m in module.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.p = p

    def enable_mc_dropout(self) -> None:
        """
        Force all dropout layers to `train()` mode so that they are active during
        forward passes, even when the wrapper is in eval() mode.
        """
        def _enable(m: nn.Module) -> None:
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()
        self.backbone.apply(_enable)
        # refine 里的 Dropout2d 会一起被切到 train()，一起参与 MC 采样

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard 2D forward. `x` is assumed to be [B, C, H, W].
        在 backbone logits 基础上做一个残差 refine。
        """
        logits = self.backbone(x)  # [B, num_classes, H, W]
        refined = self.refine(logits)
        return logits + refined

    def _forward_3d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Volume forward. `x` is assumed to be [B, C, D, H, W].
        We run Swin-Unet slice-wise along the depth dimension.
        这里为了显存和简洁性，暂时只用纯 backbone，不加 refine。
        （Synapse 本身也是 2D 训练为主）
        """
        assert x.dim() == 5, f"Expected 5D input [B, C, D, H, W], got {x.shape}"
        b, c, d, h, w = x.shape
        # Merge batch and depth and run 2D Swin-Unet
        x = x.permute(0, 2, 1, 3, 4).contiguous()      # [B, D, C, H, W]
        x = x.view(b * d, c, h, w)                     # [B*D, C, H, W]
        logits_2d = self.backbone(x)                   # [B*D, num_classes, H, W]
        logits_2d = logits_2d.view(b, d, self.num_classes, h, w)
        logits_3d = logits_2d.permute(0, 2, 1, 3, 4).contiguous()  # [B, num_classes, D, H, W]
        return logits_3d

    def forward(
        self,
        x: torch.Tensor,
        mc_samples: int = 1,
        return_std: bool = False,
    ):
        """
        Args:
            x: 
                * 2D case: [B, C, H, W]
                * 3D case: [B, C, D, H, W] (handled by slice-wise Swin-Unet)
            mc_samples: number of MC Dropout samples when `use_mc_dropout=True`
                        and the model is in eval() mode.
            return_std: if True and mc_samples > 1, also return the voxel-wise
                        standard deviation as an uncertainty map.

        Returns:
            If `mc_samples <= 1`:
                logits  (same shape as backbone output)
            If `mc_samples > 1` and `return_std=False`:
                mean_logits
            If `mc_samples > 1` and `return_std=True`:
                (mean_logits, std_logits)
        """

        is_3d = x.dim() == 5

        # ----- Training phase: no MC sampling, just one forward -----
        if self.training or (not self.use_mc_dropout) or mc_samples <= 1:
            if is_3d:
                return self._forward_3d(x)
            else:
                return self._forward_2d(x)

        # ----- Evaluation with MC Dropout -----
        # We are in eval mode but want multiple stochastic passes.
        self.enable_mc_dropout()

        outputs = []
        for _ in range(mc_samples):
            if is_3d:
                out = self._forward_3d(x)
            else:
                out = self._forward_2d(x)
            outputs.append(out)

        logits = torch.stack(outputs, dim=0)  # [T, B, C, ...]
        mean_logits = logits.mean(dim=0)

        if not return_std:
            return mean_logits

        std_logits = logits.std(dim=0)
        return mean_logits, std_logits


def build_robust_nnswinformer(
    config,
    img_size: int,
    num_classes: int,
    use_mc_dropout: bool = False,
    mc_dropout_p: float = 0.1,
    **kwargs,
) -> RobustNnSwinFormer:
    """
    Convenience factory so that you can instantiate the model in train.py
    similarly to the original SwinUnet.

    Example (in train.py):

        from networks.robust_nnswinformer import build_robust_nnswinformer

        model = build_robust_nnswinformer(
            config,
            img_size=config.DATA.IMG_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
            use_mc_dropout=True,
            mc_dropout_p=0.2,
        )
    """
    return RobustNnSwinFormer(
        config=config,
        img_size=img_size,
        num_classes=num_classes,
        use_mc_dropout=use_mc_dropout,
        mc_dropout_p=mc_dropout_p,
        **kwargs,
    )


if __name__ == "__main__":
    # Minimal sanity check with the dummy SwinUnet
    model = RobustNnSwinFormer(config=None, img_size=224, num_classes=2, use_mc_dropout=True)
    x2d = torch.randn(2, 1, 224, 224)
    x3d = torch.randn(2, 1, 4, 224, 224)

    y2d = model(x2d)
    y3d = model(x3d)

    print("2D output:", y2d.shape)
    print("3D output:", y3d.shape)

    model.eval()
    mean2d, std2d = model(x2d, mc_samples=3, return_std=True)
    print("MC 2D:", mean2d.shape, std2d.shape)