# RL_A3
项目简介

本项目实现并对比了三种经典的强化学习算法在 CartPole-v1 环境上的表现：
	•	REINFORCE（策略梯度）
	•	Actor-Critic（AC）
	•	Advantage Actor-Critic（A2C）

所有方法均在相同环境下运行，并通过多 seed 实验和插值对齐的方式进行公平比较。




环境配置

建议使用 Python 3.9+。

安装依赖：

pip install -r requirements.txt

注意：
由于 PyTorch 与 NumPy 2.x 版本可能不兼容，建议使用：

pip install numpy==1.26.4

运行方法
	1.	训练 REINFORCE

python train_reinforce.py
	2.	训练 Actor-Critic

python train_ac.py
	3.	训练 A2C

python train_a2c.py

实验设置
	•	环境：CartPole-v1
	•	每个算法使用 3 个随机种子（seeds）
	•	使用 environment steps 进行插值对齐
	•	绘制 mean + std 曲线进行对比
