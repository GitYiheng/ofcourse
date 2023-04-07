import numpy as np
import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

N_ROLLOUT = 16


def get_latest_file_path(file_dir):
    # Always get the latest file
    # https://github.com/tensorflow/tensorboard/blob/master/tensorboard/summary/writer/event_file_writer.py#L78
    file_list = os.listdir(file_dir)
    file_list.sort(key=lambda x: int(x.split(".")[3]))
    return os.path.join(file_dir, file_list[-1])


def load_data(log_path, tag=None, smooth=1):
    event_acc = EventAccumulator(log_path)
    event_acc.Reload()
    data = [scalar.value for scalar in event_acc.Scalars(tag)]
    if smooth > 1:
        """
        The formula for a simple moving average is:
            smoothed_x[t] = average(x[t-k+1], x[t-k+2], ..., x[t])
        where the "smooth" param is width of that window (k)
        """
        x = np.asarray(data)
        y = np.ones(smooth)
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, "same") / np.convolve(z, y, "same")
        data = list(smoothed_x)

    return data


def plot(log_dirs, tag="Return", smooth=1):
    epoch_list, method_list, seed_list, data_list = [], [], [], []
    xaxis = "Episode"
    plt.figure()

    if isinstance(log_dirs, str) and os.path.isdir(log_dirs):
        log_data = load_data(get_latest_file_path(log_dirs), tag=tag, smooth=smooth)
        epoch_list += [i * N_ROLLOUT for i in range(log_data.__len__())]
        seed_list += [0] * log_data.__len__()
        data_list += log_data
        data = {xaxis: epoch_list, 'Seed': seed_list, tag: data_list}
        data = pd.DataFrame(data)
        sns.lineplot(data=data, x=xaxis, y=tag)

    elif isinstance(log_dirs, list):
        for seed_idx, log_dir in enumerate(log_dirs):
            log_data = load_data(get_latest_file_path(log_dir), tag=tag, smooth=smooth)
            epoch_list += [i * N_ROLLOUT for i in range(log_data.__len__())]
            seed_list += [seed_idx] * log_data.__len__()
            data_list += log_data
        data = {xaxis: epoch_list, 'Seed': seed_list, tag: data_list}
        data = pd.DataFrame(data)
        sns.lineplot(data=data, x=xaxis, y=tag)

    elif isinstance(log_dirs, dict):
        for method in log_dirs.keys():
            assert isinstance(log_dirs[method], list)
            for seed_idx, log_dir in enumerate(log_dirs[method]):
                log_data = load_data(get_latest_file_path(log_dir), tag=tag, smooth=smooth)
                epoch_list += [i * N_ROLLOUT for i in range(log_data.__len__())]
                method_list += [method] * log_data.__len__()
                seed_list += [seed_idx] * log_data.__len__()
                data_list += log_data
        data = {xaxis: epoch_list, 'method': method_list, 'Seed': seed_list, tag: data_list}
        data = pd.DataFrame(data)
        sns.lineplot(data=data, x=xaxis, y=tag, hue="method", style="method")
    else:
        raise TypeError("Expected list, dict or str, but got %s" % type(log_dirs))

    plt.legend(loc="lower right")
    plt.grid(False)
    plt.show()
