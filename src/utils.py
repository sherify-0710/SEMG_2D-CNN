import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.preprocessing import StandardScaler


#========================================滤波函数=============================

# def apply_filter(emg_data, fs, lowcut, highcut, notch_freq=50):
#     """通用滤波函数，参数可调"""
#     fc = fs / 2
#     b_low, a_low = signal.butter(4, lowcut / fc, btype='low')
#     b_high, a_high = signal.butter(4, highcut/fc, btype='high')
#     w0 = notch_freq / fc
#     b_notch, a_notch = signal.iirnotch(w0, Q=30)
#     # 顺序滤波
#     emg_filtered = signal.filtfilt(b_high, a_high, emg_data, axis=0)
#     emg_filtered = signal.filtfilt(b_notch, a_notch, emg_filtered, axis=0)
#     emg_filtered = signal.filtfilt(b_low, a_low, emg_filtered, axis=0)
#     return emg_filtered
#================================滤波==================================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    emg_filtered_band = filtfilt(b, a, data,axis=0)
    return emg_filtered_band
# 定义50 Hz 陷波滤波器函数
def notch_filter(data, cutoff=50.0, Q=30.0, fs=200):
    nyquist = 0.5 * fs
    freq = cutoff / nyquist
    b, a = iirnotch(freq, Q)
    emg_filtered_notch = filtfilt(b, a, data,axis=0)
    return  emg_filtered_notch
#=========================================找到连续相同手势的起止索引===================
def find_contiguous_segments(restimulus, label):
    """找到连续相同手势的起止索引"""
    segments_index = []
    i = 0
    n = len(restimulus)
    while i < n:
        if restimulus[i] == label:
            start = i
            while i < n and restimulus[i] == label:
                i += 1
            end = i
            segments_index.append((start, end))
        else:
            i += 1
    return segments_index
#===========================================划分窗口===================
def sliding_window(segment, window_size, step_size):
    """将连续信号切分成固定大小的滑动窗口，这个里面传进来的segment已经是切好的了，比如就是手势6的1000个采样点，不足的直接舍弃"""
    L = segment.shape[0]
    windows = []
    for start in range(0, L - window_size + 1, step_size):
        end = start + window_size
        window = segment[start:end, :]
        windows.append(window)
    return np.array(windows)
#=======================================归一化函数=================
class ChannelwiseScaler:
    """对每个通道独立进行StandardScaler拟合和转换"""

    def __init__(self):
        self.scalers = []

    def fit(self, X):
        n_channels = X.shape[2]
        self.scalers = []
        for ch in range(n_channels):
            scaler = StandardScaler()
            channel_data = X[:, :, ch].reshape(-1, 1)
            scaler.fit(channel_data)#计算均值标准差
            self.scalers.append(scaler)
        return self

    def transform(self, X):
        X_norm = np.zeros_like(X)
        n_channels = X.shape[2]
        for ch in range(n_channels):
            channel_data = X[:, :, ch].reshape(-1, 1)
            channel_norm = self.scalers[ch].transform(channel_data)
            X_norm[:, :, ch] = channel_norm.reshape(X.shape[0], X.shape[1])
        return X_norm

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)