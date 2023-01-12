import matplotlib.pyplot as plt
import random

position = 0  # 设置初始位置
walk = []  # 保存位置
steps = 500  # 设置步数为500步
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1  # 如果随机值等于0则step为1，反之为-1
    position += step  # 改变位置（正，负）
    walk.append(position)

from scipy.signal import savgol_filter

y1 = savgol_filter(walk, 15, 11, mode="nearest")

y2 = savgol_filter(walk, 15, 1, mode="nearest")

fig = plt.figure()  # 生成窗口
ax = fig.add_subplot(211)  # 返回一个axes对象，里面的参数abc表示在一个figure窗口中，有a行b列个小窗口，然后本次plot在第c个窗口中
ax.plot(y1)

ax = fig.add_subplot(212)
ax.plot(y2)

# ax = fig.add_subplot(223)
# ax.plot(walk)

# ax = fig.add_subplot(224)
# ax.plot(walk)
plt.show()
# print walk#打印每一次的累积步数

