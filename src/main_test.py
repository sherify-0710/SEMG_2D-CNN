# main.py
#随机划分 80%+20%
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

# 导入自定义模块

from utils import  find_contiguous_segments, sliding_window, ChannelwiseScaler, butter_bandpass_filter,notch_filter
from image_transform import batch_windows_to_images, get_hilbert_coordinate
from cnn_model import SimpleCNN, train_cnn
# ===== 新增：设置中文显示 =====
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


# ---------- 数据集参数 ----------
FS = 100  # 采样率 (Hz)  ← DB1=100, DB2=2000, DB5=200
N_CHANNELS = 10  # 通道数  ← DB1=10, DB2=12, DB5=16
DATA_PATH_TEMPLATE = r"D:/PythonProject/Data/S{}_A1_E2.mat"  # 文件路径模板
TARGET_LABELS = (6, 13,14,15,16)  # 要识别的手势标签（根据你的数据修改）

# ---------- 滑动窗口参数 ----------
WINDOW_SEC = 2.56  # 窗口长度 (秒)  ← 可调，通常0.5秒
WINDOW_SIZE = int(FS * WINDOW_SEC)  # 窗口点数（自动计算）
STEP_SEC = 1.28  # 步长 (秒)  ← 可调，通常窗口长度的一半
STEP_SIZE = int(FS * STEP_SEC)  # 步长点数（自动计算）

# ---------- 图像生成参数 ----------
IMG_HEIGHT = 16  # 图像高度（点）← 需满足 H×W = WINDOW_SIZE #修改了一下，生成8*8图像
IMG_WIDTH =16   # 图像宽度（点）← 5×10=50，正好等于WINDOW_SIZE
QUANTIZE = True # 是否分层量化 ← True=用分层像素值，False=用原始幅值
N_LAYERS = 8  # 分层数/灰度级别 ← 先试16，可调4,8,16,32,64

# ---------- 训练参数 ----------
TEST_SIZE = 0.2  # 测试集比例 (0.2 = 20%)
RANDOM_STATE = 42  # 随机种子，保证结果可复现
BATCH_SIZE = 32  # 批量大小 ← 显存大可以增大，显存小可以减小
EPOCHS = 100  # 训练轮数 ← 可调，观察loss收敛后可以提前停止
LEARNING_RATE = 0.001  # 学习率 ← 可调，通常0.001
USE_INDEPENDENT_SAMPLES = False  # True=10通道作为独立样本，False=作为多通道图像,通常用10通道，这样能保留电极的相关性

# ---------- 其他 ----------
APPLY_FILTER_BAND = False  # 是否滤波（暂不启用）
APPLY_FILTER_NOTCH =False
PRINT_PROGRESS = False  # 是否打印进度
COMBINE = True #是否把原图像和分层图像融合
Dropout_rate=0.3#随机扔掉
Use_channel_attention=False
ORDER=4

#-------随机种子---------------
torch.manual_seed(RANDOM_STATE)      # 固定 CPU 和 GPU（如果使用 CUDA）的部分随机操作
np.random.seed(RANDOM_STATE)         # 固定 numpy 的随机种子

# 如果你使用 CUDA（GPU），建议再加这两行：
torch.cuda.manual_seed(RANDOM_STATE) # 固定当前 GPU 的随机操作
torch.cuda.manual_seed_all(RANDOM_STATE) # 固定所有 GPU（多卡时）
# 同时设置 cuDNN 为确定性模式（会略微降低性能，但保证完全可复现）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ==============================
# 1. 加载数据
print("=" * 50)
print("步骤1: 加载数据")
print("=" * 50)

# 受试者列表（DB1 有27个受试者，S1~S27）
SUBJECTS = list(range(1, 28))
X_all = []  # 存放所有窗口数据
y_all = []  # 存放所有窗口标签
for sub in SUBJECTS:
    #DATA_PATH_TEMPLATE = r"D:/PythonProject/Data/S{}_A1_E2.mat"  # 文件路径模板
    file_path = DATA_PATH_TEMPLATE.format(sub)#str.format() 就是：把字符串里的 {} 当成占位符，用你传进去的参数替换这些 {}，最后返回一个填好内容的新字符串。r"D:/PythonProject/Data/S{}_A1_E2.mat"
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在 {file_path}")
        continue
    # 加载.mat文件
    data = loadmat(file_path)
    emg = data['emg']  # 形状 (时间点, 10)
    restimulus = data['restimulus'].flatten()  # 形状 (时间点,)
    # 可选滤波（暂时跳过）   #自定义滤波函数
    if APPLY_FILTER_BAND:
        lowcut=5
        highcut=45#定义滤波的范围#
        emg = butter_bandpass_filter(emg,lowcut,highcut,FS,ORDER)#FS是采样率，不同数据的采样率不一样
    if APPLY_FILTER_NOTCH:
       cutoff=50
       Q=30
       emg=notch_filter(emg, cutoff,Q,FS)

    # 对每个目标手势，提取连续段的滑动窗口
    for label in TARGET_LABELS:
        segments_index = find_contiguous_segments(restimulus, label)# segments 里面存放的是】[(起始索引，终止索引)、(,)]放的是元组（整体）
        for start, end in segments_index:
            segment = emg[start:end, :]  # 取出这段信号。这一段信号是同一个手势
            # 如果这段信号长度小于窗口大小，跳过，就是不足以切割了
            if segment.shape[0] < WINDOW_SIZE:
                continue
            # 滑动窗口切分
            windows = sliding_window(segment, WINDOW_SIZE, STEP_SIZE)#windows=【窗口数量，窗口长度，通道数】windows里面放得是一段一段的window，windows是（比如手6的0~64采样点、（滑动步长32）、32~96个采样点....）
            # 添加到列表
            X_all.append(windows)
            y_all.append([label] * windows.shape[0])
    if PRINT_PROGRESS:
        print(f"受试者 {sub}: 已处理")
# 合并所有数据
if len(X_all) == 0:
    raise ValueError("没有加载到任何数据！请检查文件路径和手势标签")
X_all = np.concatenate(X_all,axis=0)  # 形状 (总窗口数, 窗口长度, 通道数)
y_all = np.concatenate(y_all)  # 形状 (总窗口数,)
print(f"\n总窗口数: {X_all.shape[0]}")
print(f"窗口形状: {X_all.shape[1]}点 × {X_all.shape[2]}通道")
print(f"手势分布: {np.unique(y_all, return_counts=True)}")
# ==============================
# 2. 划分训练集和测试集
# ==============================
print("\n" + "=" * 50)
print("步骤2: 划分训练集/测试集")
print("=" * 50)
X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(
    X_all, y_all,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
    stratify=y_all  # 保证训练集和测试集的手势比例相同
)
print(f"训练集: {X_train_raw.shape[0]} 个窗口")
print(f"测试集: {X_test_raw.shape[0]} 个窗口")
#==============================
# 3. 标准化（每个通道独立）
# ==============================
print("\n" + "=" * 50)
print("步骤3: 通道独立标准化")
print("=" * 50)
scaler = ChannelwiseScaler()
X_train_norm = scaler.fit_transform(X_train_raw)  # 训练集：拟合+转换，得到均值和标准差
X_test_norm = scaler.transform(X_test_raw)  # 测试集：只转换,归一化用的方差和均值参考训练集的，没有用测试集的，否则
#先划分训练集在归一化处理，如果先归一化处理的话，那么划分的训练集数据里面也受到了测试集数据的影响（因为方差均值是按照整体算的），就会造成数据泄露
print("标准化完成")
# ==============================
# 4. 转换为图像
# ==============================
print("\n" + "=" * 50)
print("步骤4: 生成Hilbert曲线图像")
print("=" * 50)
print(f"图像尺寸: {IMG_HEIGHT}×{IMG_WIDTH} = {IMG_HEIGHT * IMG_WIDTH}点")
print(f"是否分层量化: {'是' if QUANTIZE else '否'}")
print(f"分层数: {N_LAYERS if QUANTIZE else '不适用'}")
img_shape = (IMG_HEIGHT, IMG_WIDTH)
xx,yy,grid_size=get_hilbert_coordinate(IMG_HEIGHT ,IMG_WIDTH)
global_min = np.percentile(X_train_norm, 1)
global_max = np.percentile(X_train_norm, 99)
abs_train_all = np.abs(X_train_norm)
# 生成0, 0.125, 0.25 ... 1.0 共9个分位数点
quant_split = np.linspace(0, 1, N_LAYERS + 1)
split_thresholds = np.quantile(abs_train_all, quant_split)
print(global_min, global_max)
import numpy as np
# 1. 原始未归一化数据分布
print("========= 归一化前 X_train_raw 全局统计 =========")
print(f"全局min: {X_train_raw.min():.4f}")
print(f"全局max: {X_train_raw.max():.4f}")
print(f"整体均值: {X_train_raw.mean():.4f}")
print(f"整体标准差: {X_train_raw.std():.4f}")
# 随便取一个通道看单通道原始分布
ch0_raw = X_train_raw[:, :, 0]
print(f"\n通道0原始均值: {ch0_raw.mean():.4f}, 标准差: {ch0_raw.std():.4f}")
print(f"通道0原始一小段数据:\n{ch0_raw[0, :10]}")
# 2. 标准化
scaler = ChannelwiseScaler()
X_train_norm = scaler.fit_transform(X_train_raw)
X_test_norm = scaler.transform(X_test_raw)
print("\n========= 归一化后 X_train_norm 全局统计 =========")
print(f"全局min: {X_train_norm.min():.4f}")
print(f"全局max: {X_train_norm.max():.4f}")
print(f"整体均值: {X_train_norm.mean():.4f}")
print(f"整体标准差: {X_train_norm.std():.4f}")
# 对应通道归一化后
ch0_norm = X_train_norm[:, :, 0]
print(f"\n通道0归一后均值: {ch0_norm.mean():.4f}, 标准差: {ch0_norm.std():.4f}")
print(f"通道0归一后一小段数据:\n{ch0_norm[0, :10]}")
if COMBINE:
    print('===============原始+分层图像=================')
    X_train_img_raw = batch_windows_to_images(
        X_train_norm,
        img_shape=img_shape,
        quantize=False,
        n_layers=None,
        split_thresh=split_thresholds,
        xx=xx,
        yy=yy,
        grid_size=grid_size
    )
    X_test_img_raw = batch_windows_to_images(
        X_test_norm,
        img_shape=img_shape,
        quantize=False,
        n_layers=None,
        split_thresh=split_thresholds,
        xx = xx,
        yy = yy,
        grid_size = grid_size
    )
    # 生成分层图像（离散值）
    X_train_img_quant = batch_windows_to_images(
        X_train_norm,
        img_shape=img_shape,
        quantize=True,  # 量化
        n_layers=N_LAYERS,  # 比如 16
        split_thresh=split_thresholds,
        xx = xx,
        yy = yy,
        grid_size = grid_size
    )
    X_test_img_quant = batch_windows_to_images(
        X_test_norm,
        img_shape=img_shape,
        quantize=True,
        n_layers=N_LAYERS,
        split_thresh=split_thresholds,
        xx=xx,
        yy=yy,
        grid_size=grid_size
    )
    X_train_img = np.concatenate([X_train_img_raw, X_train_img_quant], axis=1)#在通道维度进行拼接
    X_test_img = np.concatenate([X_test_img_raw, X_test_img_quant], axis=1)
    print(f"原始图像形状: {X_train_img_raw.shape}")
    print(f"分层图像形状: {X_train_img_quant.shape}")
    print(f"拼接后形状: {X_train_img.shape}")
# else 必须和对应的 if 对齐，不能缩进
else:
    print(f'不启动融合,是否分层：',QUANTIZE)
    X_train_img = batch_windows_to_images(
        X_train_norm,
        img_shape=img_shape,
        quantize=QUANTIZE,
        n_layers=N_LAYERS
    )
    X_test_img = batch_windows_to_images(
        X_test_norm,
        img_shape=img_shape,
        quantize=QUANTIZE,
        n_layers=N_LAYERS
    )
    print(f"\n训练图像形状: {X_train_img.shape}")  # (n_windows, 10, H, W)
    print(f"测试图像形状: {X_test_img.shape}")
in_channels = X_train_img.shape[1]

# 1. 取第一个窗口、第一个通道的Hilbert图像（浮点数幅值）
first_window_first_ch = X_train_img[100, 10, :, :]  # 形状 ，1078
# 2. 绘制图像（适配浮点数幅值的显示）
first2_window_first_ch = X_train_img[100, 0, :, :]  # 形状 ，1078
# 同一份样本，对比原始 / 量化通道
idx = 1078
raw_ch = X_train_img[idx, 0, :, :]    # 原始浮点图（0~9通道）
quant_ch = X_train_img[idx, 10, :, :] # 量化分层图（10~19通道）

print("原始浮点通道唯一值数量：", len(np.unique(raw_ch)))
print("量化分层通道唯一值：", np.unique(quant_ch))

plt.figure(figsize=(8, 4))
binary_img = (first_window_first_ch > 0).astype(float)#######
# 用灰度图显示，vmin/vmax设为±3（标准化后EMG幅值的典型范围）
im = plt.imshow(first_window_first_ch, cmap='gray')
plt.colorbar(im, label='EMG幅值（标准化浮点数）')
plt.title('第1078个窗口-通道 10的Hilbert曲线图像')
plt.xlabel('图像宽度（像素）')
plt.ylabel('图像高度（像素）')
plt.xticks(range(15))  # 显示宽度像素索引（0~9）
plt.yticks(range(15))   # 显示高度像素索引（0~4）
plt.grid(alpha=0.3, linestyle='--')  # 加网格更易看像素位置


plt.figure(figsize=(8, 4))
binary_img = (first2_window_first_ch > 0).astype(float)#######
# 用灰度图显示，vmin/vmax设为±3（标准化后EMG幅值的典型范围）
im = plt.imshow(first2_window_first_ch, cmap='gray')
plt.colorbar(im, label='EMG幅值（标准化浮点数）')
plt.title('第1078个窗口-通道0的Hilbert曲线图像')
plt.xlabel('图像宽度（像素）')
plt.ylabel('图像高度（像素）')
plt.xticks(range(15))  # 显示宽度像素索引（0~9）
plt.yticks(range(15))   # 显示高度像素索引（0~4）
plt.grid(alpha=0.3, linestyle='--')  # 加网格更易看像素位置

# 3. 保存图像到本地（方便查看）
# 关键修改：加transparent=False，关闭透明通道
plt.savefig(
    r"D:/PythonProject/Data/hilbert_image_example.png",
    dpi=150,
    bbox_inches='tight',
    transparent=False  # 新增这一行，彻底解决警告
)
plt.show()
print("\n=== 量化效果诊断 ===")
sample_img = X_train_img[0, 0, :, :]
print(f"数据类型: {sample_img.dtype}")
print(f"唯一值数量: {len(np.unique(sample_img))}")
if QUANTIZE:
    print(f"期望唯一值 ≤ {N_LAYERS}, 实际唯一值: {np.unique(sample_img)}")
else:
    print(f"期望大量浮点值, 实际前10个唯一值: {np.unique(sample_img)[:10]}")
# 4. 打印该图像的关键数值信息（验证）
print("\n=== 可视化图像数值信息 ===")
print(f"图像形状: {first_window_first_ch.shape}")
print(f"幅值范围: {first_window_first_ch.min():.4f} ~ {first_window_first_ch.max():.4f}")
print(f"幅值均值: {first_window_first_ch.mean():.4f}")
print(f"幅值标准差: {first_window_first_ch.std():.4f}")
##标签编码，原始标签→0-4连续整数
print("\n" + "=" * 50)
print("步骤4.5: 标签编码（原始标签→0-4连续整数）")
print("=" * 50)
from sklearn.preprocessing import LabelEncoder
# 初始化标签编码器
le = LabelEncoder()
# 拟合+转换训练集标签（6,13,14,15,16 → 0,1,2,3,4）
y_train_encoded = le.fit_transform(Y_train)
# 仅转换测试集标签（复用训练集的映射规则，避免数据泄露）
y_test_encoded = le.transform(Y_test)
# 打印标签映射关系（方便后续解读结果）
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"标签编码映射: {label_mapping}")  # 输出：{6:0, 13:1, 14:2, 15:3, 16:4}
print(f"训练集标签范围: {y_train_encoded.min()} ~ {y_train_encoded.max()}")  # 0~4
print(f"测试集标签范围: {y_test_encoded.min()} ~ {y_test_encoded.max()}")  # 0~4


# ==============================
# 5. 准备CNN输入数据
# ==============================
print("\n" + "=" * 50)
print("步骤5: 准备CNN输入")
print("=" * 50)
if USE_INDEPENDENT_SAMPLES:
    # 方案A: 10个通道作为独立样本（样本数 × 10）
    print("使用方案A: 10个通道作为独立样本")
    n_train = X_train_img.shape[0]
    n_test = X_test_img.shape[0]
    n_channels = X_train_img.shape[1]
    # 展平通道维度，每个通道变成独立样本
    X_train_cnn = X_train_img.reshape(-1, 1, IMG_HEIGHT, IMG_WIDTH)
    X_test_cnn = X_test_img.reshape(-1, 1, IMG_HEIGHT, IMG_WIDTH)
    # 标签重复10次
    y_train_cnn = np.repeat(y_train_encoded, n_channels)
    y_test_cnn = np.repeat(y_test_encoded, n_channels)
    in_channels = 1
    print(f"训练样本数: {X_train_cnn.shape[0]} (原窗口数×{n_channels})")
    print(f"测试样本数: {X_test_cnn.shape[0]}")
else:
    # 方案B: 10个通道作为多通道图像（类似RGB的10通道版本）
    print("使用方案B: 10个通道作为多通道图像")
    # 调整维度顺序： (n_windows, channels, H, W) 已经是正确格式
    X_train_cnn = X_train_img  # 形状 (n_train, 10, H, W)
    X_test_cnn = X_test_img  # 形状 (n_test, 10, H, W)
    y_train_cnn = y_train_encoded
    y_test_cnn = y_test_encoded
# if COMBINE:
#     in_channels = N_CHANNELS*2#乘2是融合后
# else:
#     in_channels = N_CHANNELS#乘2是融合后
    print(f"训练样本数: {X_train_cnn.shape[0]}")
    print(f"测试样本数: {X_test_cnn.shape[0]}")
    print(f"输入通道数: {in_channels}")
# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train_cnn, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_cnn, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_cnn, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_cnn, dtype=torch.long)
# 创建DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"\nDataLoader创建完成")
print(f"训练集batch数: {len(train_loader)}")
print(f"测试集batch数: {len(test_loader)}")
# ==============================
# 6. 初始化CNN模型
# ==============================
print("\n" + "=" * 50)
print("步骤6: 初始化CNN模型")
print("=" * 50)
# 确定分类数
num_classes = len(TARGET_LABELS)#目标手势数量
print(f"分类数: {num_classes} (手势 {TARGET_LABELS})")
# 选择设备（GPU或CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
# 步骤6初始化模型后补充
model = SimpleCNN(
    in_channels=in_channels,
    num_classes=num_classes,
    img_size=(IMG_HEIGHT, IMG_WIDTH),
    dropout_rate = Dropout_rate,
    use_channel_attention = Use_channel_attention
)
model = model.to(device)  # 新增这行，显式把模型移到GPU（避免隐性报错）
torch.cuda.empty_cache()  # 清空GPU缓存，训练更稳定
#==============================
# 7. 训练CNN
# ==============================
print("\n" + "=" * 50)
print("步骤7: 训练CNN")
print("=" * 50)
# 使用我们之前写的训练函数
final_accuracy = train_cnn(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    device=device
)



