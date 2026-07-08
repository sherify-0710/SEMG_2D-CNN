import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from skimage.transform import resize

xuanshenme=1;
#-----对数-----
def quantize_signal0(signal, n_layers):
    """
    方案3：对数量化
    使用对数压缩，更适合肌电信号的动态范围
    """
    # 取绝对值（如果信号可能为负）
    signal_abs = np.abs(signal)

    # 对数压缩：log(1 + x * scale_factor)
    # scale_factor 控制压缩程度，可根据信号范围调整
    scale_factor = 100  # 经验值，可根据实际信号调整
    log_signal = np.log1p(signal_abs * scale_factor)

    min_val = log_signal.min()
    max_val = log_signal.max()

    if max_val == min_val:
        return np.zeros_like(signal, dtype=int)

    scaled = (log_signal - min_val) / (max_val - min_val)
    quantized = np.floor(scaled * (n_layers - 1)).astype(int)

    return quantized

# #===============全局线性量化===========
# def quantize_global_linear(signal, n_layers, global_min, global_max):
#     """
#     全局线性量化：使用固定的 min/max（不依赖窗口），先裁剪再线性映射到 [0, n_layers-1]
#     """
#     if global_max == global_min:
#         return np.zeros_like(signal, dtype=int)
#     # 裁剪到全局范围
#     clipped = np.clip(signal, global_min, global_max)
#     # 线性拉伸
#     scaled = (clipped - global_min) / (global_max - global_min)
#     # 量化
#     quantized = np.floor(scaled * (n_layers - 1)).astype(int)
#     return quantized
def quantize_global_linear(signal, n_layers, global_min, global_max):
    # 方案一核心：先取绝对值，丢弃负相位信息
    signal_abs = np.abs(signal)
    # 绝对值信号最小值天然为0，重新限定有效映射区间 [0, global_max]
    lower = 0.0
    upper = global_max
    if upper == lower:
        return np.zeros_like(signal_abs, dtype=int)
    # 仅裁剪非负幅值
    clipped = np.clip(signal_abs, lower, upper)
    # 线性均分 [0, global_max] 区间
    scaled = (clipped - lower) / (upper - lower)
    quantized = np.floor(scaled * (n_layers - 1)).astype(int)
    # 防止浮点误差溢出层级
    quantized = np.clip(quantized, 0, n_layers - 1)
    return quantized


def quantize_equal_freq(signal, split_thresh, n_layers):
    signal_abs = np.abs(signal)
    quant_res = np.zeros_like(signal_abs, dtype=int)
    # 遍历8个层级赋值
    for level in range(n_layers):
        left = split_thresh[level]
        right = split_thresh[level + 1]
        mask = (signal_abs >= left) & (signal_abs <= right)
        quant_res[mask] = level
    return quant_res
#===============================方案一===================
#每个窗口独立，保证充分利用全部灰度级
#不同窗口间相同幅值可能映射到不同灰度值
# def quantize_signal1(signal, n_layers):
#     """
#     将一维信号幅值分层量化（连续值 → 离散层级）
#     【参数说明】
#     signal : 一维数组，长度 L，比如一个窗口的某个通道数据
#     n_layers : 分层数/灰度级别（如 4, 8, 16, 32...）
#     【返回值】
#     quantized : 一维整数数组，每个值在 0 ~ n_layers-1 之间
#     """
#     min_val = signal.min()
#     max_val = signal.max()
#     if max_val == min_val:
#         return np.zeros_like(signal, dtype=int)
#     scaled = (signal - min_val) / (max_val - min_val)
#     quantized = np.floor(scaled * (n_layers - 1)).astype(int)
#     return quantized
def get_hilbert_coordinate(h,w):
    """
    输入信号长度L，输出坐标xx, yy与网格尺寸
    整个程序只调用这一次
    """
    L=h* w
    p = 1
    while (2 ** (2 * p)) < L:
        p += 1
    grid_size = 2 ** p
    hc = HilbertCurve(p, 2)
    coords = hc.points_from_distances(np.arange(L))
    coords = np.array(coords)
    xx = coords[:, 0]
    yy = coords[:, 1]
    return xx, yy, grid_size
def signal_to_hilbert_image(signal, img_shape, quantize, n_layers, split_thresh,xx,yy,grid_size):
    """
    将一维信号转换为二维Hilbert曲线图像索引
    【参数说明】
    signal : 一维数组，长度 L，#第i个窗口第ch通道的信号（长度是窗口长度哦）在下面那个函数中调用了
    img_shape : 目标图像尺寸 (高度, 宽度)，必须满足 高度×宽度 = L
    quantize : 是否先分层量化（True/False）
    n_layers : 如果 quantize=True，分层数（灰度级别）

    【返回值】
    image : 二维数组，形状 = img_shape
    """
    L = len(signal)
    H, W = img_shape
    assert H * W == L, f"图像尺寸{H}×{W}={ H *W}必须等于信号长度{L}"
    if quantize:
        if xuanshenme:
                    #signal = quantize_global_linear(signal, n_layers,global_min, global_max)
                    signal = quantize_equal_freq(signal, split_thresh, n_layers)
        else:
            signal= quantize_signal0(signal, n_layers)
    # 确定Hilbert曲线阶数p
    temp = np.zeros((grid_size, grid_size), dtype=signal.dtype)
    temp[xx, yy] = signal
#缩放到目标尺寸
    image = resize(temp.astype(float), (H, W), order=0, preserve_range=True).astype(np.int32)#signal.dtype
    return image
    #image 确实是一个形状为 (H, W) 的二维数组，里面填充的是采样点的幅值，填充顺序按照希尔伯特曲线的规则。
def batch_windows_to_images(X, img_shape, quantize, n_layers,split_thresh,xx,yy,grid_size):  #def batch_windows_to_images(X, img_shape, quantize, n_layers, return_both=False):
    """
    将批量窗口数据转换为图像，所以每个通道的window_len个时间点，被reshape成H×W的图像。
    【参数说明】
    X : 形状 (n_windows, window_len, n_channels)
    img_shape : 目标图像尺寸 (H, W)
    quantize : 是否先分层量化
    n_layers : 如果 quantize=True，分层数de
    【返回值】
    images : 形状 (n_windows, n_channels, H, W)
    """
    n_windows, window_len, n_channels = X.shape
    H, W = img_shape
    assert H * W == window_len, f"图像尺寸{H}×{W}={ H *W}必须等于窗口长度{window_len}"
    images = np.zeros((n_windows, n_channels, H, W), dtype=np.float32)
    for i in range(n_windows):
        for ch in range(n_channels):
            signal = X[i, :, ch]#第i个窗口第ch通道的信号（长度是窗口长度哦）
            img = signal_to_hilbert_image(
                signal,
                img_shape,
                quantize,
                n_layers,
                #global_min,
                #global_max,
                split_thresh,
                xx,
                yy,
                grid_size
            )
            images[i, ch, :, :] = img

        if (i + 1) % 100 == 0:
            print(f"  图像生成进度: { i +1}/{n_windows}")

    return images
