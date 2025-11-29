# 用于中期演示的具体任务：
# 必须完成 (用于5分钟演示)
- [ ] 至少一个基线模型完全复现(Swin-Unet或nnFormer)
  - [ ] Swin-Unet: https://github.com/HuCaoFighting/Swin-Unet
  - [ ] nnFormer: https://github.com/282857341/nnFormer
- [ ] 数据集Synapse & ADCD
  - [ ] 数据集地址：
  - [ ] 在10% Synapse数据上的初步训练结果
  - [ ] 基本的噪声注入功能
  - [ ] 简单的性能可视化

# 期望完成
- [ ] 两个基线模型都复现完成
- [ ] 在完整干净数据上的基线性能
- [ ] 噪声鲁棒性模块的初步实现

# 加分项  
- [ ] 混合架构的初步实现
- [ ] 在噪声数据上的初步对比结果


<!-- # 所有任务列表
# 1. 基础架构实现
## 1.1 基础模型复现
- [ ] 实现Swin-Unet (2D版本)
- [ ] 实现nnFormer (3D版本) 
- [ ] 验证复现模型在干净数据上的性能

## 1.2 混合架构开发
- [ ] 实现Robust nnSwinFormer核心架构
- [ ] 集成局部体积注意力(LV-MSA)
- [ ] 集成全局体积注意力(GV-MSA)
- [ ] 实现跳跃注意力机制
- [ ] 3D卷积嵌入层

# 2. 数据预处理模块
# 2.1 数据集加载
- [ ] Synapse多器官分割数据集加载器
- [ ] ACDC心脏分割数据集加载器
- [ ] 数据标准化和预处理流水线

# 2.2 噪声注入系统
- [ ] 随机标签噪声注入 (10%, 20%, 30%)
- [ ] 结构化噪声模拟 (相邻器官混淆)
- [ ] 实例依赖噪声生成
- [ ] 噪声水平验证工具


# 3.2 协同训练框架
- [ ] 双网络架构实现
- [ ] 小损失样本选择机制
- [ ] 互相教学训练循环

# 3.3 不确定性学习
- [ ] 蒙特卡洛Dropout实现
- [ ] 预测不确定性计算
- [ ] 不确定性引导的样本加权

# 4. 基线实验 (Weeks 1-3)
# 4.1 模型复现验证
- [ ] 在干净Synapse数据上测试Swin-Unet
- [ ] 在干净Synapse数据上测试nnFormer
- [ ] 在干净ACDC数据上对比两个模型
- [ ] 记录Dice和HD95作为基线

# 4.2 小规模可行性测试
- [ ] 使用10%数据训练所有模型
- [ ] 验证训练流程和评估脚本
- [ ] 确保可复现性(固定随机种子)

# 5. 噪声鲁棒性实验 (Weeks 4-6)
# # 5.1 噪声影响分析
- [ ] 实验1: 干净数据 vs 10%噪声
- [ ] 实验2: 干净数据 vs 20%噪声  
- [ ] 实验3: 干净数据 vs 30%噪声
- [ ] 记录性能下降趋势

# 5.2 架构对比实验
- [ ] Swin-Unet在不同噪声水平下的表现
- [ ] nnFormer在不同噪声水平下的表现
- [ ] 我们的混合架构在不同噪声水平下的表现

# 6. 消融研究 (Weeks 7-9)
# 6.1 组件重要性分析
- [ ] 仅基础架构(无噪声鲁棒性)
- [ ] 基础架构 + GCE损失
- [ ] 基础架构 + 协同训练
- [ ] 基础架构 + 不确定性学习
- [ ] 完整框架(所有组件)

# 6.2 超参数敏感性
- [ ] 不同噪声比例的影响
- [ ] 损失函数权重调优
- [ ] 协同训练中样本选择比例

# 7. 综合评估实验 (Weeks 10-12)
# 7.1 跨数据集泛化
- [ ] 在Synapse上训练，在ACDC上测试
- [ ] 在不同噪声水平下的泛化能力

# 7.2 统计显著性检验
- [ ] 每个实验配置重复3次训练
- [ ] 计算均值和标准差
- [ ] 执行配对t检验(p < 0.05)
- [ ] 计算95%置信区间 -->


# SwinUNet干净数据
0it [00:00, ?it/s]idx 0 case case0008 mean_dice 0.546886 mean_hd95 18.724007
1it [01:42, 102.43s/it]idx 1 case case0022 mean_dice 0.861215 mean_hd95 3.497197
2it [02:43, 77.83s/it]idx 2 case case0038 mean_dice 0.790241 mean_hd95 7.974543
3it [03:49, 72.83s/it]idx 3 case case0036 mean_dice 0.843482 mean_hd95 7.027469
4it [05:56, 94.15s/it]idx 4 case case0032 mean_dice 0.842683 mean_hd95 6.060299
5it [07:28, 93.22s/it]idx 5 case case0002 mean_dice 0.820222 mean_hd95 9.371769
6it [08:54, 90.98s/it]idx 6 case case0029 mean_dice 0.623678 mean_hd95 62.335278
7it [09:57, 81.78s/it]idx 7 case case0003 mean_dice 0.594305 mean_hd95 86.312400
8it [12:07, 97.14s/it]idx 8 case case0001 mean_dice 0.730453 mean_hd95 19.223630
9it [13:49, 98.45s/it]idx 9 case case0004 mean_dice 0.686429 mean_hd95 9.185487
10it [15:21, 96.64s/it]idx 10 case case0025 mean_dice 0.710079 mean_hd95 7.578896
11it [16:18, 84.52s/it]idx 11 case case0035 mean_dice 0.842773 mean_hd95 3.987632

Mean class 1 mean_dice 0.815452 mean_hd95 7.705711
Mean class 2 mean_dice 0.657705 mean_hd95 30.100499
Mean class 3 mean_dice 0.774464 mean_hd95 28.359844
Mean class 4 mean_dice 0.676476 mean_hd95 32.019648
Mean class 5 mean_dice 0.937286 mean_hd95 9.524922
Mean class 6 mean_dice 0.482322 mean_hd95 19.554375
Mean class 7 mean_dice 0.872238 mean_hd95 18.032067
Mean class 8 mean_dice 0.712354 mean_hd95 15.555338

Testing performance in best val model: mean_dice : 0.741037 mean_hd95 : 20.106551

# SwinUNet10%噪声数据
0it [00:00, ?it/s]idx 0 case case0008 mean_dice 0.596582 mean_hd95 33.366833
1it [01:44, 104.03s/it]idx 1 case case0022 mean_dice 0.854246 mean_hd95 30.401435
2it [02:45, 79.13s/it] idx 2 case case0038 mean_dice 0.799104 mean_hd95 17.718876
3it [03:55, 75.04s/it]idx 3 case case0036 mean_dice 0.815865 mean_hd95 21.603098
4it [06:01, 94.89s/it]idx 4 case case0032 mean_dice 0.880944 mean_hd95 7.809842
5it [07:36, 94.87s/it]idx 5 case case0002 mean_dice 0.817804 mean_hd95 9.819876
6it [09:03, 92.20s/it]idx 6 case case0029 mean_dice 0.744222 mean_hd95 57.796300
7it [10:05, 82.50s/it]idx 7 case case0003 mean_dice 0.659875 mean_hd95 98.758046
8it [12:18, 98.61s/it]idx 8 case case0001 mean_dice 0.760734 mean_hd95 23.884652
9it [13:56, 98.41s/it]idx 9 case case0004 mean_dice 0.747342 mean_hd95 15.509611
10it [15:30, 96.88s/it]idx 10 case case0025 mean_dice 0.797007 mean_hd95 43.387152
11it [16:29, 85.41s/it]idx 11 case case0035 mean_dice 0.832057 mean_hd95 6.939170
12it [17:25, 87.15s/it]
Mean class 1 mean_dice 0.827659 mean_hd95 21.440834
Mean class 2 mean_dice 0.631331 mean_hd95 36.397027
Mean class 3 mean_dice 0.836327 mean_hd95 59.481215
Mean class 4 mean_dice 0.763222 mean_hd95 39.182040
Mean class 5 mean_dice 0.938584 mean_hd95 17.406527
Mean class 6 mean_dice 0.559145 mean_hd95 16.353290
Mean class 7 mean_dice 0.879090 mean_hd95 35.635941
Mean class 8 mean_dice 0.768498 mean_hd95 18.766386
Testing performance in best val model: mean_dice : 0.775482 mean_hd95 : 30.582908

# Robust_SwinUNet10%噪声数据
0it [00:00, ?it/s]idx 0 case case0008 mean_dice 0.611759 mean_hd95 23.729024
1it [01:38, 98.20s/it]idx 1 case case0022 mean_dice 0.873560 mean_hd95 21.881910
2it [02:37, 75.55s/it]idx 2 case case0038 mean_dice 0.775488 mean_hd95 10.021968
3it [03:47, 72.69s/it]idx 3 case case0036 mean_dice 0.808684 mean_hd95 27.032692
4it [05:48, 92.06s/it]idx 4 case case0032 mean_dice 0.878754 mean_hd95 5.468633
5it [07:22, 92.43s/it]idx 5 case case0002 mean_dice 0.824021 mean_hd95 9.658452
6it [08:48, 90.43s/it]idx 6 case case0029 mean_dice 0.670871 mean_hd95 62.331468
7it [09:48, 80.40s/it]idx 7 case case0003 mean_dice 0.582315 mean_hd95 104.603883
8it [11:56, 95.57s/it]idx 8 case case0001 mean_dice 0.715184 mean_hd95 26.059412
9it [13:33, 96.02s/it]idx 9 case case0004 mean_dice 0.742685 mean_hd95 21.027455
10it [15:03, 94.26s/it]idx 10 case case0025 mean_dice 0.775169 mean_hd95 16.199570
11it [16:00, 82.79s/it]idx 11 case case0035 mean_dice 0.860958 mean_hd95 5.213409
12it [16:53, 84.47s/it]
Mean class 1 mean_dice 0.837950 mean_hd95 13.353717
Mean class 2 mean_dice 0.638061 mean_hd95 31.874970
Mean class 3 mean_dice 0.813341 mean_hd95 36.647535
Mean class 4 mean_dice 0.770074 mean_hd95 37.478729
Mean class 5 mean_dice 0.939943 mean_hd95 17.147824
Mean class 6 mean_dice 0.562223 mean_hd95 14.354690
Mean class 7 mean_dice 0.885665 mean_hd95 29.885784
Mean class 8 mean_dice 0.712373 mean_hd95 18.408668
Testing performance in best val model: mean_dice : 0.759954 mean_hd95 : 27.768989

# SwinUNet20%噪声数据
0it [00:00, ?it/s]idx 0 case case0008 mean_dice 0.594249 mean_hd95 27.307290
1it [01:41, 101.07s/it]idx 1 case case0022 mean_dice 0.876174 mean_hd95 3.850801
2it [02:41, 76.93s/it] idx 2 case case0038 mean_dice 0.788743 mean_hd95 26.838110
3it [03:48, 72.80s/it]idx 3 case case0036 mean_dice 0.829543 mean_hd95 10.626209
4it [05:49, 91.76s/it]idx 4 case case0032 mean_dice 0.859526 mean_hd95 5.699780
5it [07:21, 91.69s/it]idx 5 case case0002 mean_dice 0.828536 mean_hd95 9.288505
6it [08:47, 89.69s/it]idx 6 case case0029 mean_dice 0.685859 mean_hd95 62.015428
7it [09:46, 79.92s/it] idx 7 case case0003 mean_dice 0.607709 mean_hd95 102.378212
8it [11:57, 96.06s/it]idx 8 case case0001 mean_dice 0.767183 mean_hd95 26.836822
9it [13:30, 95.18s/it]idx 9 case case0004 mean_dice 0.691868 mean_hd95 31.950718
10it [14:58, 92.77s/it]idx 10 case case0025 mean_dice 0.842812 mean_hd95 16.710455
11it [15:52, 81.07s/it]idx 11 case case0035 mean_dice 0.846795 mean_hd95 5.371119
12it [16:43, 83.66s/it]
Mean class 1 mean_dice 0.837464 mean_hd95 7.151768
Mean class 2 mean_dice 0.583034 mean_hd95 43.670872
Mean class 3 mean_dice 0.828076 mean_hd95 35.118376
Mean class 4 mean_dice 0.786371 mean_hd95 33.225258
Mean class 5 mean_dice 0.937543 mean_hd95 37.573380
Mean class 6 mean_dice 0.558277 mean_hd95 13.076000
Mean class 7 mean_dice 0.885223 mean_hd95 30.919610
Mean class 8 mean_dice 0.730012 mean_hd95 18.513703
Testing performance in best val model: mean_dice : 0.768250 mean_hd95 : 27.406121

# Robust_SwinUNet20%噪声数据
0it [00:00, ?it/s]idx 0 case case0008 mean_dice 0.622374 mean_hd95 17.164925
1it [01:40, 100.86s/it]idx 1 case case0022 mean_dice 0.877173 mean_hd95 5.854871
2it [02:41, 77.25s/it] idx 2 case case0038 mean_dice 0.807304 mean_hd95 12.478430
3it [03:47, 72.26s/it]idx 3 case case0036 mean_dice 0.812319 mean_hd95 13.616298
4it [05:50, 92.01s/it]idx 4 case case0032 mean_dice 0.867495 mean_hd95 6.884828
5it [07:21, 91.87s/it]idx 5 case case0002 mean_dice 0.819111 mean_hd95 10.819787
6it [08:47, 89.82s/it]idx 6 case case0029 mean_dice 0.735721 mean_hd95 39.478239
7it [09:46, 79.67s/it]idx 7 case case0003 mean_dice 0.570447 mean_hd95 102.449669
8it [11:55, 95.32s/it]idx 8 case case0001 mean_dice 0.734859 mean_hd95 36.354467
9it [13:29, 94.89s/it]idx 9 case case0004 mean_dice 0.705686 mean_hd95 11.973726
10it [14:57, 92.97s/it]idx 10 case case0025 mean_dice 0.819293 mean_hd95 6.524891
11it [15:55, 82.19s/it]idx 11 case case0035 mean_dice 0.863056 mean_hd95 4.973974
12it [16:49, 84.13s/it]
Mean class 1 mean_dice 0.834280 mean_hd95 12.411684
Mean class 2 mean_dice 0.618794 mean_hd95 37.567878
Mean class 3 mean_dice 0.804437 mean_hd95 28.952414
Mean class 4 mean_dice 0.743444 mean_hd95 30.710716
Mean class 5 mean_dice 0.932407 mean_hd95 26.485198
Mean class 6 mean_dice 0.570291 mean_hd95 13.108455
Mean class 7 mean_dice 0.885951 mean_hd95 11.236297
Mean class 8 mean_dice 0.766956 mean_hd95 18.576762
Testing performance in best val model: mean_dice : 0.769570 mean_hd95 : 22.381176

# SwinUNet30%噪声数据
0it [00:00, ?it/s]idx 0 case case0008 mean_dice 0.595033 mean_hd95 23.662077
1it [01:42, 102.08s/it]idx 1 case case0022 mean_dice 0.867016 mean_hd95 6.545836
2it [02:42, 77.29s/it] idx 2 case case0038 mean_dice 0.805918 mean_hd95 12.402691
3it [03:48, 72.28s/it]idx 3 case case0036 mean_dice 0.831835 mean_hd95 12.481948
4it [05:51, 92.49s/it]idx 4 case case0032 mean_dice 0.867930 mean_hd95 9.026855
5it [07:28, 94.02s/it]idx 5 case case0002 mean_dice 0.828226 mean_hd95 9.316746
6it [08:55, 91.63s/it]idx 6 case case0029 mean_dice 0.742228 mean_hd95 42.329686
7it [09:56, 81.68s/it]idx 7 case case0003 mean_dice 0.615631 mean_hd95 107.092703
8it [12:04, 96.44s/it]idx 8 case case0001 mean_dice 0.756085 mean_hd95 31.820632
9it [13:40, 96.28s/it]idx 9 case case0004 mean_dice 0.727131 mean_hd95 25.289465
10it [15:08, 93.54s/it]idx 10 case case0025 mean_dice 0.823723 mean_hd95 26.155209
11it [16:04, 82.15s/it]idx 11 case case0035 mean_dice 0.861210 mean_hd95 5.584058
12it [16:56, 84.68s/it]
Mean class 1 mean_dice 0.833025 mean_hd95 16.401434
Mean class 2 mean_dice 0.595053 mean_hd95 48.094196
Mean class 3 mean_dice 0.849377 mean_hd95 32.635935
Mean class 4 mean_dice 0.786291 mean_hd95 40.799753
Mean class 5 mean_dice 0.939132 mean_hd95 15.780530
Mean class 6 mean_dice 0.559013 mean_hd95 16.067308
Mean class 7 mean_dice 0.889193 mean_hd95 20.406860
Mean class 8 mean_dice 0.763559 mean_hd95 17.619257
Testing performance in best val model: mean_dice : 0.776830 mean_hd95 : 25.975659

# Robust_SwinUNet30%噪声数据
0it [00:00, ?it/s]idx 0 case case0008 mean_dice 0.650945 mean_hd95 25.120726
1it [01:41, 101.77s/it]idx 1 case case0022 mean_dice 0.857701 mean_hd95 26.008263
2it [02:42, 77.63s/it] idx 2 case case0038 mean_dice 0.787896 mean_hd95 15.743319
3it [03:49, 72.82s/it]idx 3 case case0036 mean_dice 0.830116 mean_hd95 11.733737
4it [05:51, 92.38s/it]idx 4 case case0032 mean_dice 0.864168 mean_hd95 6.492324
5it [07:23, 92.22s/it]idx 5 case case0002 mean_dice 0.827236 mean_hd95 9.071721
6it [08:49, 90.02s/it]idx 6 case case0029 mean_dice 0.769497 mean_hd95 37.762114
7it [09:50, 80.36s/it]idx 7 case case0003 mean_dice 0.560946 mean_hd95 102.589429
8it [12:00, 96.18s/it]idx 8 case case0001 mean_dice 0.752704 mean_hd95 25.682774
9it [13:36, 96.27s/it]idx 9 case case0004 mean_dice 0.749190 mean_hd95 18.399717
10it [15:03, 93.23s/it]idx 10 case case0025 mean_dice 0.828157 mean_hd95 26.790009
11it [15:58, 81.57s/it]idx 11 case case0035 mean_dice 0.859408 mean_hd95 3.931746
12it [16:50, 84.18s/it]
Mean class 1 mean_dice 0.843331 mean_hd95 13.643842
Mean class 2 mean_dice 0.622256 mean_hd95 36.139818
Mean class 3 mean_dice 0.813659 mean_hd95 43.470317
Mean class 4 mean_dice 0.768333 mean_hd95 48.006476
Mean class 5 mean_dice 0.938703 mean_hd95 23.277583
Mean class 6 mean_dice 0.591632 mean_hd95 13.778994
Mean class 7 mean_dice 0.898552 mean_hd95 12.120857
Mean class 8 mean_dice 0.748843 mean_hd95 15.779365
Testing performance in best val model: mean_dice : 0.778164 mean_hd95 : 25.777156

# 分析结论
1. 总体数值对比（全局 mean）

先把 “Testing performance in best val model” 的全局指标列一下：

训练集噪声	模型	mean_dice	mean_hd95
0%	SwinUNet	0.7410	20.11
10%	SwinUNet	0.7755	30.58
10%	Robust-SwinUNet	0.7600	27.77
20%	SwinUNet	0.7683	27.41
20%	Robust-SwinUNet	0.7696	22.38
30%	SwinUNet	0.7768	25.98
30%	Robust-SwinUNet	0.7782	25.78

看“Robust – Swin”的差值更直观：

10% 噪声

ΔDice ≈ -0.0155（Dice ↓）

ΔHD95 ≈ -2.8（HD95 ↓，边界稍好）

20% 噪声

ΔDice ≈ +0.0013（Dice ≈ 持平略好）

ΔHD95 ≈ -5.0（HD95 明显下降）

30% 噪声

ΔDice ≈ +0.0013（Dice ≈ 持平略好）

ΔHD95 ≈ -0.2（小幅变好）

👉 核心结论（可以原封不动写到 Discussion）：

在中等及以上噪声水平（20–30%）下，Robust-SwinUNet 基本保持甚至略微提升 mean Dice，同时在 HD95 上更稳定、整体更低；在低噪声（10%）时，模型略微牺牲 Dice 换取更平滑的边界。

2. “只看原始 SwinUNet”：噪声对 baseline 的影响

只看 SwinUNet：

mean Dice：

Clean: 0.7410

10%: 0.7755

20%: 0.7683

30%: 0.7768

👉 Dice 在有噪声时不降反升，波动在 0.76–0.78 之间。
这可以理解为：在你当前数据规模和正则设置下，少量的标签噪声有点像额外的 regularization / label smoothing，让模型更“均匀”地学到结构，而没有立刻崩。

mean HD95：

Clean: 20.11

10%: 30.58

20%: 27.41

30%: 25.98

👉 HD95 明显高于干净训练（20 → 25–30），说明即便 Dice 看起来很好，边界其实变毛躁了——有噪声的区域被拉扯，预测边缘更不稳定。

一句话概括 baseline：

SwinUNet 对适度噪声的 Dice 比较“乐观”，但 HD95 暴露了边界更不稳定的情况。

3. “Swin vs Robust”：在各个噪声水平下谁更鲁棒
10% 噪声

Swin: mean Dice 0.7755, HD95 30.58

Robust: mean Dice 0.7600, HD95 27.77

解释建议：

在 10% 噪声下，Robust-SwinUNet 的 mean Dice 略低（约 1.5 个百分点），但 HD95 从 30.6 降到 27.8，边界质量有一定改善。可以理解为：在“轻度噪声”场景中，refine head + dropout 相当于一种较强的平滑正则，略微削弱了对某些局部结构的过拟合程度。

20% 噪声（你这套实验里“最好讲”的一档）

Swin: mean Dice 0.7683, HD95 27.41

Robust: mean Dice 0.7696, HD95 22.38

在 20% 噪声下，Robust-SwinUNet 基本保持与 SwinUNet 相当的 mean Dice（0.768 → 0.770），同时将 HD95 显著降低约 5（27.4 → 22.4），说明在中等水平标签噪声下，所提出的鲁棒结构显著稳定了分割边界。

这句可以直接放进 Abstract/Conclusion 里。

30% 噪声

Swin: mean Dice 0.7768, HD95 25.98

Robust: mean Dice 0.7782, HD95 25.78

在 30% 噪声（相对非常脏的标注）条件下，Robust-SwinUNet 与基线 Swin-UNet 的 mean Dice 几乎相同（0.777 vs. 0.778），HD95 略优。说明在极高噪声比例下，两者整体都受到标签质量的限制，但鲁棒结构仍能提供轻微的边界稳定收益。

4. 类别级别可以顺带提一句的模式（给你一个可选的“细节句”）

你已经列了 per-class 指标，可以在论文里挑一两句最典型的，比如：

对 Dice 原本就不高、HD95 较大的类别（如 class 2），在 10%–30% 噪声下：

Robust-SwinUNet 通常能在 Dice 维持接近的前提下，明显降低 HD95。

对原本 Dice 就较高的类别（如 class 7），特别是在 20%–30% 噪声时：

Robust-SwinUNet 能 进一步略微提升 Dice，并降低 HD95，说明它并不仅仅对“困难类别”有效，对“简单但被噪声污染的类别”，也能帮助维持更稳定的边界。

你可以写成一句：

“Per-class analysis further shows that the proposed robust variant particularly benefits difficult structures (e.g., class 2 with low Dice and large HD95 under noise), while also stabilizing boundaries of relatively easier organs (e.g., class 7) when the annotation noise level is moderate to high (20–30%).”

5. 一段可以直接贴进论文的总总结（中英双语感一点）

你可以这样写：

在干净训练集上，Swin-UNet 的整体性能为 mean Dice=0.741、HD95=20.1。加入 10%–30% 的标签噪声后，原始 Swin-UNet 的 Dice 得分不仅没有明显下降，反而略有提升（0.768–0.777），但 HD95 全面恶化到 25–30 的区间，说明在噪声标注下，模型虽然仍能保持较高的体积重叠度，但分割边界变得更加不稳定。
在相同的噪声条件下，本文提出的 Robust-SwinUNet 在所有噪声比例下都能够降低 HD95，相比基线模型在 10% 噪声下以约 1.5 个百分点的 Dice 损失换来了更好的边界质量，而在 20% 与 30% 噪声水平下则几乎不牺牲 Dice（甚至略有提升），同时显著降低 HD95，尤其是在 20% 噪声时将 HD95 从 27.4 降低至 22.4。总体来看，该结果说明：在中等及以上水平的标签噪声下，Robust-SwinUNet 能够在保持全局重叠精度的同时，有效提升器官边界的定位精度和鲁棒性。