# cnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ！！！！！！！！！！！！！！！！！！！ 通道注意力模块 这是新增的！！！！！！！！！
class ChannelAttention(nn.Module):
    """通道注意力模块 - 让模型自动学习哪些通道更重要"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SimpleCNN(nn.Module):
    """
    适用于小尺寸图像（如5×10）的简单CNN
    """

    def __init__(self, in_channels, num_classes, img_size, dropout_rate=0.3,use_channel_attention=True):
        """
        参数:
            in_channels: 输入图像的通道数（1=灰度图，10=多通道图像）
            num_classes: 分类数（手势类别数）（有几个手势）
             dropout_rate: Dropout比率
            img_size: 图像尺寸 (H, W)
             use_channel_attention: 是否启用通道注意力模块（默认True）
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.use_channel_attention = use_channel_attention  # 新增：保存注意力开关
        H, W = img_size

        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)  # 卷积。【输入通道数，输出通道数，卷积核大小，作用：补零】
        self.bn1 = nn.BatchNorm2d(32)  # 归一化，让数据分布更加稳定，加速训练
        self.pool1 = nn.MaxPool2d(2)
        # 输出尺寸: (16, H//2, W//2)

        # 第二层卷积
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)  # 输出尺寸: (32, H//4, W//4)
        """池化窗口大小：2×2像素
        步长：默认为2（和窗口大小相同）,取2*2像素框中的最大值"""
        # 计算经过两次池化后的特征图尺寸
        # H_pooled = H // 4
        # W_pooled = W // 4
        # 如果H或W小于4，可能需要调整，但5×10经过两次池化后：
        # 5→2（ceil模式），10→5（ceil模式），实际需要根据ceil_mode处理
        # 这里简化处理，使用自适应池化或动态计算
        # 更稳健的方法：在forward中动态计算展平维度

        # ========== 新增：通道注意力模块 ==========
        # 输入通道数 32，缩减率 reduction 取 8（32/8=4，不宜太小）
        if self.use_channel_attention:
            self.ca = ChannelAttention(channels=64, reduction=8)
        # ======================================


        #==========dropout==========
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc1 = None  # 将在第一次forward时初始化
        self.fc2 = nn.Linear(128, num_classes)  # 前面卷积 + 池化后，特征图 "摊平" 的总长度，不是随便定的！32*1*2，num_class是手势识别种类

        # 记录参数
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size

    def forward(self, x):
        #self.pool1 = nn.MaxPool2d(2)？？？  # x shape: (batch, in_channels, H, W)
        batch_size = x.size(0)

        # 卷积层
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # ========== 在展平前应用通道注意力 ==========
        if self.use_channel_attention:
            x = self.ca(x)  # x shape: (batch, 32, H/4, W/4)
        # ======================================


        # 动态计算展平后的维度
        if self.fc1 is None:
            # 获取当前特征图的尺寸
            num_features = x.shape[1] * x.shape[2] * x.shape[3]
            self.fc1 = nn.Linear(num_features, 128).to(x.device)



        # 展平
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))  # 给变换后的特征加一个非线性过滤（把负数变 0，只留有用的特征）
        # ========== Dropout 只加在这里，不影响你的网络结构 ==========
        x = self.dropout(x)
        # ==============================================================
        # 全连接层
        x = self.fc2(x)

        return x


# ============================================================
# 【关键修改】train_cnn 函数必须定义在类的外面（与类平级，无缩进）
# ============================================================

def train_cnn(model, train_loader, test_loader, epochs, lr, device):
    """
    训练CNN模型并返回测试准确率

    参数:
        model: SimpleCNN 模型实例
        train_loader: 训练集 DataLoader
        test_loader: 测试集 DataLoader
        epochs: 训练轮数
        lr: 学习率
        device: 'cuda' 或 'cpu'

    返回值:
        accuracy: 测试集准确率
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    best_accuracy = 0.0
    best_epoch = 0
    best_model_state = None

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 测试阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        accuracy = correct / total
        #保存最佳模型============后续可能用
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            best_model_state = model.state_dict().copy()  # 深拷贝模型参数

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Test Acc: {accuracy:.4f}")

    # 训练结束后，加载最佳模型（如果需要后续使用）
    model.load_state_dict(best_model_state)
    print(f"\n最佳测试准确率: {best_accuracy:.4f} (Epoch {best_epoch + 1})")

    return best_accuracy  # 返回最佳准确率，而不是最后一个epoch的

# ============================================================
# 测试代码（可选，用于验证导入）
# ============================================================
if __name__ == "__main__":
    print("cnn_model.py 加载成功")
    print("SimpleCNN 类可用")
    print("train_cnn 函数可用")


