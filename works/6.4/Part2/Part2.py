import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.models import resnet50
from torchvision import models
from PIL import Image
import os
import random
from shutil import copyfile

# 定义数据集路径和类别标签
data_dir = './data_classification'
class_labels = ['apple', 'banana', 'corn', 'durian', 'peanut']

# 定义图像预处理转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像张量
])

# 定义训练集和测试集的比例和随机种子
train_ratio = 0.6  # 训练集比例
random_seed = 42  # 随机种子

# 创建训练集和测试集的文件夹
train_dir = './train_data'
test_dir = './test_data'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 划分数据集
for class_label in class_labels:
    class_dir = os.path.join(data_dir, class_label)
    train_class_dir = os.path.join(train_dir, class_label)
    test_class_dir = os.path.join(test_dir, class_label)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    image_files = os.listdir(class_dir)
    random.seed(random_seed)
    random.shuffle(image_files)

    num_train = int(train_ratio * len(image_files))
    train_files = image_files[:num_train]
    test_files = image_files[num_train:]

    for train_file in train_files:
        src_path = os.path.join(class_dir, train_file)
        dst_path = os.path.join(train_class_dir, train_file)
        copyfile(src_path, dst_path)

    for test_file in test_files:
        src_path = os.path.join(class_dir, test_file)
        dst_path = os.path.join(test_class_dir, test_file)
        copyfile(src_path, dst_path)

# 加载训练集和测试集
train_dataset = DatasetFolder(train_dir, loader=Image.open, extensions='.jpg', transform=transform)
test_dataset = DatasetFolder(test_dir, loader=Image.open, extensions='.jpg', transform=transform)

# 创建数据加载器
batch_size = 32
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型架构
model = resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_labels))

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
torch.cuda.empty_cache()
num_epochs = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for images, labels in train_data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_data_loader)
    print(f'Training - Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Testing - Accuracy: {accuracy:.4f}')

# 保存模型
torch.save(model.state_dict(), 'model.pth')
