import numpy as np
import matplotlib.pyplot as plt
import random

from OneEuroFilter import OneEuroFilter


def sin_wave(A, f, fs, phi, t):
    """
    :params A:    振幅
    :params f:    信号频率
    :params fs:   采样频率
    :params phi:  相位
    :params t:    时间长度
    """
    # 若时间序列长度为 t=1s,
    # 采样频率 fs=1000 Hz, 则采样时间间隔 Ts=1/fs=0.001s
    # 对于时间序列采样点个数为 n=t/Ts=1/0.001=1000, 即有1000个点,每个点间隔为 Ts
    Ts = 1 / fs
    n = t / Ts
    n = np.arange(n)
    y = A * np.sin(2 * np.pi * f * n * Ts + phi * (np.pi / 180))
    return y


signal = sin_wave(0.3, 1.0, 60, 0, 10)

# mu: m
mu = 0
# std: m (from real data)
sigma = 0.01
# speed: m/s
# speed = 0.4
# sample_dt: s
sample_dt = 0.01667

data_real = []

data_sim = []

for i in range(100):
    data_sim.append(signal[0] + random.gauss(mu, sigma))
    data_real.append(signal[0])

for i in range(600):
    data_real.append(signal[i])
    x_err = signal[i] + random.gauss(mu, sigma)
    data_sim.append(x_err)

for i in range(100):
    data_real.append(signal[-1])
    data_sim.append(signal[-1] + random.gauss(mu, sigma))

fig = plt.figure()  # 生成窗口
ax = fig.add_subplot(211)
ax.plot(data_real)
# ax = fig.add_subplot(312)
# ax.plot(data_sim)


f = OneEuroFilter(signal[0], 0.0, 0.001, 20.0, 3.0)

data_filtered = []
data_filtered.append(signal[0])
for i in range(1, len(data_sim)):
    filtered_ = f(i * sample_dt, data_sim[i])
    data_filtered.append(filtered_)

# ax = fig.add_subplot(313)
# ax.plot(data_filtered)

fig.align_labels()
plt.tight_layout()
plt.show()

data_real_np = np.array(data_real)
data_sim_np = np.array(data_sim)
data_filtered_np = np.array(data_filtered)

print(np.mean(data_sim_np - data_real_np))
print(np.mean(data_filtered_np - data_real_np))

print(np.std(data_sim_np - data_real_np))
print(np.std(data_filtered_np - data_real_np))
