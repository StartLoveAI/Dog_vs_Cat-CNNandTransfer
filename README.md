# 项目

基于基础的CNN和迁移学习分别实现的kaggle猫狗图片分类

## 说明

### 环境配置

```html
# GPU云服务器跑
```

### 数据集

Kaggle比赛项目：https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition

- 数据集为：25000张猫狗数据
- 测试集为：12500张猫狗数据

> 数据集可以在Kaggle找到下载！

### 仓库

本仓库包括以下：

- `basic_cnn_model.py`：基础的CNN结构进行的猫狗识别分类；
- `transfer_model.py`：迁移学习进行的猫狗识别分类；
- `use_model_to_test.py`：模型训练完成后用于测试的程序；

**注意**：迁移学习之前需要导入一些预训练好的模型（例如：[inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5](https://download.csdn.net/download/gilgame/10970073) ）

> 基础的CNN和迁移学习完成后都会生成一个`.h5`的模型文件，在进行测试之前将`.h5`模型文件导入到测试程序即可！

## 使用

运行项目：
```python
# 运行cnn训练模型
python basic_cnn_model.py

# 运行迁移学习训练模型
python transfer_model.py

# 进行模型测试
python use_model_to_test.py
```

## CNN架构

<center><img src="https://s1.ax1x.com/2020/09/10/wJwGXd.png" alt="wJwGXd.png" border="0" /></center>


