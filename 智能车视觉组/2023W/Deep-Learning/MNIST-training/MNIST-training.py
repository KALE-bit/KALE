import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 1024)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义训练和测试函数
def train(net, device, train_loader, criterion, optimizer, epoch):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 计算准确率和损失
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        train_loss = running_loss / (batch_idx + 1)
        train_accuracy = 100.0 * correct / total

        # 输出进度条
        progress_bar.set_postfix(loss=train_loss, accuracy=train_accuracy)
        progress_bar.update(1)

        # 检查是否达到目标准确率
        if train_accuracy >= 98.0:
            flag = True
            break
        else:
            flag = False

    return flag

def test(net, device, test_loader):
    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = net(data)
            loss = criterion(outputs, labels)

            # 计算准确率和损失
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss /= len(test_loader)
            test_accuracy = 100.0 * correct / total
            # 输出进度条
            progress_bar.set_postfix(test_loss=test_loss, test_accuracy=test_accuracy)
            progress_bar.update(1)

        return test_accuracy


# 定义超参数
batch_size = 128
learning_rate = 0.01
num_epochs = 20
# 加载数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 初始化神经网络、损失函数和优化器
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

# 定义进度条
train_iter = len(train_loader.dataset) // batch_size
test_iter = len(test_loader.dataset) // batch_size
progress_bar = tqdm(total=train_iter * num_epochs + test_iter)

def output():
    # 训练神经网络
    flag = False
    for epoch in range(num_epochs):
        train_flag = train(net, device, train_loader, criterion, optimizer, epoch)
        if train_flag:
            flag = True
            break
        else:
            flag = False

    # 测试神经网络
    test_accuracy = test(net, device, test_loader)
    print('Test Accuracy: %.2f %%' % (test_accuracy))

    # 检查是否达到目标准确率
    if flag and test_accuracy >= 98.0:
        print('Achieved target accuracy of 98% on test set!')
    else:
        output()

output()
