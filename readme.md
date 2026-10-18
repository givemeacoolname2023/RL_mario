## 简介

基于stablebaseline3强化学习框架和gym-super-mario-bros马里奥游戏包，使用PPO算法, 训练马里奥通关。

本项目是按照b站主播【算法路飞】的教程 https://www.bilibili.com/video/BV1CERYY3EjA/?p=2&share_source=copy_web&vd_source=dbb60edcfbcee053b3e3e7aa16ec24be 完成的项目，但实际代码和教程中的略有不同。

在原本的框架基础上，修改了奖励函数机制，调整了超参数，目前500w步的训练可以通关第一关。

## 运行方法

1、环境（Python 3.8.8）

```
conda create -n RL_mario python=3.8 -y
conda activate RL_mario
```

2、依赖

```python
pip install torch  # 我选择的是torch-2.2.2+cu121-cp38-cp38-win_amd64.whl，建议选择合适的torch版本下载到本地，再进行安装
# 有3个要单独安装的旧版本库
pip install setuptools==65.5.0
pip install wheel==0.38.4
pip install pip==20.2.4
# 完整下载依赖
pip install -r requirements.txt
```

3、训练

```
python train_model.py
```

4、查看训练曲线

```
tensorboard --logdir logs
```

5、测试效果

```
python test_model.py
```