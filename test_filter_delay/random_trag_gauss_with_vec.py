import numpy as np
import matplotlib.pyplot as plt
import random

from OneEuroFilter import OneEuroFilter

from scipy.interpolate import InterpolatedUnivariateSpline

def find_interpolate_argmax(x_line, y_line):
    f = InterpolatedUnivariateSpline(x_line, y_line, k=4)
    cr_pts = f.derivative().roots() 
    cr_pts = np.append(cr_pts, (y_line[0], y_line[-1])) # also check the endpoints of the interval
    cr_vals = f(cr_pts)
    min_index = np.argmin(cr_vals)
    max_index = np.argmax(cr_vals)
    # print("Maximum value {} at {}\nMinimum value {} at {}".format(cr_vals[max_index],cr_pts[max_index],cr_vals[min_index],cr_pts[min_index]))
    return cr_pts[max_index]


def get_latency_ms(A, f = 0.8):


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
    
    
    signal = sin_wave(A, f, 60, 0, 10)
    
    # mu: m
    mu = 0
    # std: m (from real data)
    sigma = 0.5
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
    
    data_real_np = np.array(data_real)
    data_sim_np = np.array(data_sim)
    
    filter_pos = OneEuroFilter(data_real_np[0], 0.0, 0.001, 10.0, 1.0)
    filter_vec = OneEuroFilter(data_real_np[0], 0.0, 0.01, 10.0, 3.0)
    
    data_filtered = []
    data_filtered.append(signal[0])
    for i in range(1, len(data_real)):
        filtered_ = filter_pos(i * sample_dt, data_real_np[i])
        vec = filtered_ - data_real_np[i - 1]
        vec_filtered = filter_vec(i * sample_dt, vec)
        filtered_ = filtered_ + (vec_filtered - vec)
        data_filtered.append(filtered_)
    
    data_filtered_np = np.array(data_filtered)
    
    # fig = plt.figure()  # 生成窗口
    # ax = fig.add_subplot(311)
    # ax.plot(data_real)
    # ax = fig.add_subplot(312)
    # ax.plot(data_sim)
    
    # print(np.mean(np.abs(data_real_np[100:-100] - data_filtered_np[100:-100])))
    
    corr = np.correlate(data_real_np[100:-100], data_filtered_np[100:-100], mode="same")
    max_ind = find_interpolate_argmax(np.array(range(600)), corr)
    # print("smoothed output with prediction v.s no smoothed output with prediction latency:", (len(corr) // 2 - max_ind) * sample_dt, "s")
    
    # return (len(corr) // 2 - max_ind) * sample_dt * 1000
    return (np.mean(np.abs(data_real_np[400:-100] - data_filtered_np[400:-100])))

if __name__ == "__main__":
    
    speeds = np.linspace(0.01, 5.0, 500)

    delay = []
    for i in speeds:
        delay.append(get_latency_ms(i * 0.8))
    print(get_latency_ms(i))
    fig, ax = plt.subplots()
    plt.title("speed - time delay")
    plt.xlabel("speed (m/s)")
    plt.ylabel("Pos Err (m)")
    ax.plot(speeds, delay)
    plt.show()
