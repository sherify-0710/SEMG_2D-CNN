# ablation_layers.py 单独测试不同量化分层数的影响
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from image_transform import batch_windows_to_images, get_hilbert_coordinate
from cnn_model import SimpleCNN, train_cnn

# 1. 读取主程序预处理好的数据，不用重复加载mat、标准化
with open("exp_data_cache.pkl", "rb") as f:
    cache = pickle.load(f)

# 还原全局变量
X_train_norm = cache["X_train_norm"]
X_test_norm = cache["X_test_norm"]
y_train_encoded = cache["y_train_encoded"]
y_test_encoded = cache["y_test_encoded"]
IMG_HEIGHT = cache["IMG_HEIGHT"]
IMG_WIDTH = cache["IMG_WIDTH"]
BATCH_SIZE = cache["BATCH_SIZE"]
EPOCHS = cache["EPOCHS"]
LEARNING_RATE = cache["LEARNING_RATE"]
Dropout_rate = cache["Dropout_rate"]
RANDOM_STATE = cache["RANDOM_STATE"]
num_classes = cache["num_classes"]
device = cache["device"]

# 固定实验参数
N_RUNS = 5    # 每组重复3次，提速；需要精准改为5
# 待测试的分层数量
layer_candidates = [4, 8, 16, 32]
# 两种输入模式：仅分层 / 原图+分层融合
test_modes = [
    ("only_quant", False, "仅量化分层单输入"),
    (True, True, "原始+量化双通道融合")
]

# ====================== 复用你写好的数据集构建函数 ======================
def build_dataset(combine_mode, n_layers):
    img_shape = (IMG_HEIGHT, IMG_WIDTH)
    xx, yy, grid_size = get_hilbert_coordinate(IMG_HEIGHT, IMG_WIDTH)
    abs_train_all = np.abs(X_train_norm)
    quant_split = np.linspace(0, 1, n_layers + 1)
    split_thresholds = np.quantile(abs_train_all, quant_split)

    if combine_mode is True:
        # 融合20通道
        X_train_raw = batch_windows_to_images(
            X_train_norm, img_shape, quantize=False, n_layers=None,
            split_thresh=split_thresholds, xx=xx, yy=yy, grid_size=grid_size
        )
        X_train_quant = batch_windows_to_images(
            X_train_norm, img_shape, quantize=True, n_layers=n_layers,
            split_thresh=split_thresholds, xx=xx, yy=yy, grid_size=grid_size
        )
        X_train_img = np.concatenate([X_train_raw, X_train_quant], axis=1)

        X_test_raw = batch_windows_to_images(
            X_test_norm, img_shape, quantize=False, n_layers=None,
            split_thresh=split_thresholds, xx=xx, yy=yy, grid_size=grid_size
        )
        X_test_quant = batch_windows_to_images(
            X_test_norm, img_shape, quantize=True, n_layers=n_layers,
            split_thresh=split_thresholds, xx=xx, yy=yy, grid_size=grid_size
        )
        X_test_img = np.concatenate([X_test_raw, X_test_quant], axis=1)
        in_chan = 20
    elif combine_mode == "only_quant":
        # 仅分层10通道
        X_train_img = batch_windows_to_images(
            X_train_norm, img_shape, quantize=True, n_layers=n_layers,
            split_thresh=split_thresholds, xx=xx, yy=yy, grid_size=grid_size
        )
        X_test_img = batch_windows_to_images(
            X_test_norm, img_shape, quantize=True, n_layers=n_layers,
            split_thresh=split_thresholds, xx=xx, yy=yy, grid_size=grid_size
        )
        in_chan = 10
    else:
        raise ValueError("combine_mode 仅支持 True / 'only_quant'")

    X_train_cnn = X_train_img
    X_test_cnn = X_test_img
    y_train_cnn = y_train_encoded
    y_test_cnn = y_test_encoded
    return X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn, in_chan

def run_single_exp(use_att, combine, n_layers, n_runs):
    accuracies = []
    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn, in_chan = build_dataset(combine, n_layers)

    X_train_tensor = torch.tensor(X_train_cnn, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_cnn, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_cnn, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_cnn, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for run in range(n_runs):
        seed = RANDOM_STATE + run
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)

        model = SimpleCNN(
            in_channels=in_chan,
            num_classes=num_classes,
            img_size=(IMG_HEIGHT, IMG_WIDTH),
            dropout_rate=Dropout_rate,
            use_channel_attention=use_att
        ).to(device)

        print(f"\n分层数{n_layers} | 注意力{use_att} | 输入模式{combine} 第{run+1}/{n_runs}轮")
        acc = train_cnn(model, train_loader, test_loader, EPOCHS, LEARNING_RATE, device)
        accuracies.append(acc)
        print(f"本轮Acc: {acc:.4f}")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    return mean_acc, std_acc

# ====================== 批量遍历所有分层、有无注意力 ======================
total_result = {}
for mode_flag, desc_att, mode_name in test_modes:
    for layer_num in layer_candidates:
        for att_flag in [False, True]:
            exp_key = f"{mode_name} | 分层{layer_num} | 注意力{att_flag}"
            print("\n" + "="*70)
            print(f"开始实验：{exp_key}")
            print("="*70)
            mean, std = run_single_exp(use_att=att_flag, combine=mode_flag, n_layers=layer_num, n_runs=N_RUNS)
            total_result[exp_key] = (mean, std)

# ====================== 汇总打印 ======================
print("\n" + "="*90)
print("不同量化分层数消融实验汇总（均值±标准差）")
print("="*90)
for name, (mean, std) in total_result.items():
    print(f"{name:<55} | 平均Acc: {mean:.4f} ± {std:.4f}")

# 分层数增益对比（固定输入、固定注意力）
print("\n=== 分层数量横向对比 ===")
# 举个例子：仅分层、无注意力
print("【仅量化分层、无注意力】不同分层精度：")
for l in layer_candidates:
    k = f"仅量化分层单输入 | 分层{l} | 注意力False"
    m, _ = total_result[k]
    print(f"分层{l:2d}层 -> Acc {m:.4f}")
print("="*90)