import numpy as np
import matplotlib.pyplot as plt
import random

from OneEuroFilter import OneEuroFilter

# sample_dt: s
sample_dt = 0.01667


def get_time_delay(speed, filter_params):

    """
    speed: m/s
    """

    data_real = []

    for i in range(100):
        data_real.append(0)

    for i in range(600):
        x = i * speed * sample_dt
        data_real.append(x)

    for i in range(100):
        data_real.append(x)

    data_real_np = np.array(data_real)

    f = OneEuroFilter(
        data_real_np[0], 0.0, filter_params[0], filter_params[1], filter_params[2]
    )
    f = OneEuroFilter(
        data_real_np[0], 0.0, filter_params[0], filter_params[1], filter_params[2]
    )
    
    # if speed == 0.01:
    #     fig = plt.figure()  # 生成窗口
    #     ax = fig.add_subplot(211)
    #     ax.plot(data_real)
# 
    data_filtered = []
    data_filtered.append(data_real_np[0])
    for i in range(1, len(data_real)):
        filtered_ = f(i * sample_dt, data_real_np[i])
        data_filtered.append(filtered_)

    data_filtered_np = np.array(data_filtered)

    # return (data_real_np[699] - data_filtered_np[699]) / speed * 1000
    return (data_real_np[699] - data_filtered_np[699])


if __name__ == "__main__":

    speeds = np.linspace(0.01, 5.0, 500)
    delay = []
    for i in speeds:
        delay.append(get_time_delay(i, (0.001, 10.0, 1.0)))
    print(get_time_delay(0.01, (1.0, 20.0, 3.0)))
    fig, ax = plt.subplots()
    plt.title("speed - time delay")
    plt.xlabel("speed (m/s)")
    plt.ylabel("Pos Err(m)")
    ax.plot(speeds, delay)
    plt.show()
