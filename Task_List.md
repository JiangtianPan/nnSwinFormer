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
Mean class 3 mean_dice 0.783341 mean_hd95 36.647535
Mean class 4 mean_dice 0.760074 mean_hd95 52.478729
Mean class 5 mean_dice 0.929943 mean_hd95 23.147824
Mean class 6 mean_dice 0.532223 mean_hd95 14.354690
Mean class 7 mean_dice 0.885665 mean_hd95 29.885784
Mean class 8 mean_dice 0.712373 mean_hd95 20.408668
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