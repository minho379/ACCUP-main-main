# 论文总结

## 论文研究问题

时间序列数据与其他数据类型相比提出了独特的挑战，特别是由于幅度变异性和固有噪声。幅度变异性是由于操作条件、传感器校准或个体差异等因素导致时间序列数据幅度发生变化时产生的。此外，现实世界的时间序列数据通常是嘈杂和非平稳的，外部干扰或环境因素掩盖了真实模式，使准确的数据解释复杂化。

## 解决方案

* 通过不确定性感知的原型集成，巧妙地利用增强技术和熵筛选来应对幅度变异和噪声，生产出高质量、可靠的伪标签；

* 将获取到的伪标签输入到增强对比聚类中，以一种对噪声不敏感的方式优化模型，确保伪标签被高效、安全地使用，使得模型学习到的特征在类别上的区分度更高，聚类更紧密，从而更好地捕捉细微的时序模式差异；

* 第一个专门为广泛的时间序列应用（如人体活动识别、机械故障诊断、睡眠阶段分期）设计和系统化解决TTA问题的工作，填补了该领域的空白

## 方法流程图

# 论文公式及程序代码对照

公式编号 | 所属文件 | 行数
---- | ---- | ----
(1) | accup.py | 40-41
(2) | accup.py | 84-85
(3) | accup.py | 82-88
(4) | accup.py | 56-62
(5) | sup_contrast_loss.py | 4-26
 
# 安装说明
* 克隆项目仓库
```
git clone https://github.com/Tokenmw/ACCUP-main 
```
* 根据项目README文件说明配置环境并下载数据集
* 在项目根目录下创建data文件夹，从数据集中选择.pt文件加入data文件夹

## 数据集下载
使用作者提供的三个公开数据集：
- [SSC](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/UD1IM9)
- [UCIHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ)
- [MFD](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/PU85XN)

## 环境依赖
* Python3
* Pytorch==1.9
* Numpy==1.23.5
* scikit-learn==1.0
* Pandas==1.3.4
* skorch==0.10.0
* openpyxl==3.0.7
* Wandb==0.12.7

## 运行命令
训练模型：

```
python trainers/tta_trainer.py --exp_name All_trg --da_method ACCUP --dataset HAR --backbone CNN --num_runs 3 
```

# 运行结果
