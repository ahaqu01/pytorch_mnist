# 1 加载必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 2 定义超参数
BATCH_SIZE = 512 #每批处理的数据
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #是否用GPU还是CPU训练
EPOCHS = 10 #训练数据集的轮次

# 3 构建pipeline,对图像做处理
pipeline = transforms.Compose([
    transforms.ToTensor(),#将 图片转换成tensor
    transforms.Normalize((0.1307,), (0.3081,)), #正则化，模型出现过拟合现象时，降低模型复杂度
])

# 4 下载、加载数据
from torch.utils.data import DataLoader

# 下载数据集
train_set = datasets.MNIST("data",download=True, transform=pipeline)

test_set = datasets.MNIST("data", download=True, transform=pipeline)

# 加载数据
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 插入代码，显示MNIST中的图片
with open("./data/MNIST/raw/train-images-idx3-ubyte", "rb") as f:
    file = f.read()

image1 = [int(str(item).encode('ascii'), 16) for item in file[16 : 16+784]]
print(image1)

import cv2
import numpy as np

image1_np = np.array(image1, dtype=np.uint8).reshape(28,28,1)
print(image1_np)
cv2.imwrite("digit.jpg", image1_np)

# 5 构建网络模型
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5) # 定义一个卷积层，1:灰度图片的通道， 10：输出通道， 5：kernel 5x5
        self.conv2 = nn.Conv2d(10, 20, 3) # 定义一个卷积层， 10：输入通道，20：输出通道，3：kernel
        self.fc1 = nn.Linear(20*10*10, 500) # 定义一个全连接层, 20*10*10：输入通道，500：输出通道
        self.fc2 = nn.Linear(500, 10) # 定义一个全连接层，500：输入通道，10输出通道(有10个数字)
        #print(" in Digit __init__ ok \n" )

    def forward(self, x): # 定义一个全向传播
        input_size = x.size(1) # batch_size x 1 x 28 x 28 ,28是图片的像素
        #print("batch_size = {} \t input_size = {}".format(BATCH_SIZE, input_size) )
        x = self.conv1(x) # 输入：batch_size *1*28*28,1:对应的conv1的灰度图片的通道,输出：batch_size*10*24*24 ,10:对应的conv1的输出通道，24：28-5+1（其中，5为kernel）
        x = F.relu(x) # 激活函数，保持shape不变，输出：batch_size*10*24*24，作用：在所有的隐藏层之间添加一个激活函数，这样的输出就是一个非线性函数，因而神经网络的表达能力更加强大了。
        x = F.max_pool2d(x, 2, 2) # 池化层，输入：batch_size*10*24*24，输出：batch_size*10*12*12 (2,2=>减半) ，作用：对图片进行压缩（降采样）的一种方法，如max pooling（最大池化层：比如在地图上看到一片地区，选择最大/最重要的城市）, average pooling等
        #print(" in Digit before  self.conv2(x) ok \n")
        x = self.conv2(x) # 输入：batch_size*10*12*12,10:conv2的输入通道；输出：batch_size*20*10*10,20:conv2的输出通道，10:12-3+1 (其中，3为kernel)
        x = F.relu(x) # 激活函数，保持shape不变

        x = x.view(-1,20*10*10) #拉伸或拉平，把多维的拉成一排，对应手写体网络图中的flatten。-1：自动计算维度。20*10*10=2000
        #print(" in Digit before  self.fc1(x) ok \n")
        x = self.fc1(x) # 全连接层，输入：batch_size*2000, 输出：batch_size*500 , 500：对应fc1的输出通道
        # print(" in Digit after  self.fc1(x) ok \n")
        x = F.relu(x) # 激活函数，保持shape不变

        x = self.fc2(x) # 全连接层，输出：batch_size*500, 输出：batch_size*10
        #print(" in Digit after  self.fc2(x) ok \n")
        # 计算10个类，每个类的概率是多少，给一张图片，上面有10个数字，要返回一个概率，也就是0-9的概率
        output = F.log_softmax(x, dim=1) # 也算一个损失函数，计算分类后，每个数字的概率值，并返回概率最高的数字。dim是维度的意思，和opencv一致，dim=1表示按行计算
        #print(" in Digit after  output ok \n")

        return output

# 6 定义优化器
model = Digit().to(DEVICE) # 创建一个模型，把它部署到我们的DEVICE上，如果是GPU就部署到GPU上，CPU就部署到CPU上

optimzer = optim.Adam(model.parameters()) # 创建一个优化器，作用就是更新模型的参数，使得训练和测试的结果达到一个最优值，除了Adam，还有其他优化器

state = {
    'state':model.state_dict(),
    'batch_size':BATCH_SIZE,
    'epoch':EPOCHS
}

torch.save(state, './mnist_model.pth')

# 7 定义训练方法
def train_model(model, device, train_loader, optimizer, epoch): # model训练的模型， device设备， train_loader训练的数据，optimizer优化器， epoch训练的轮次
    # 模型训练
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader): # data就是图片数据，target就是标签
        # 把模型部署到DEVICE上去
        data, target = data.to(DEVICE), target.to(DEVICE)
        # 梯度初始化为0
        optimzer.zero_grad()
        # 训练后的结果
        output = model(data)
        # 计算损失，什么是损失，损失说白了就是一个差距，比如计算了一个结果，和真实值target做一个比较，计算之间差距有多远，如果差距很大，那么证明训练的效果不好，
        # 如果预测的值和真实的值越来越靠近，就证明这个模型特别好。会把所有的损失都累加起来。是模型效果的一个指标
        loss = F.cross_entropy(output, target) # cross_entropy:交叉熵损失，针对一个多分类的任务，比如本项目0-9就适合使用这个损失函数。
                                               # 如果是二分类，我们就要使用sigmod损失函数，output:预测值，target:真实值
        # 找到概率值最大的下标,删掉了
        #pred = output.data.max(1)[1]
        #pred = output.max(1, keepdim=True) # 1：表示横轴。函数会返回每个元素概率最大值的下标。也可以写成 pred = output.argmax(dim=1)
        # 反向传播
        loss.backward()
        # 参数优化（更新）
        optimzer.step()
        if batch_idx % 3000 == 0: # 设置没3000个轮回就打印一次结果，batch_idx为60000
            print("Train Eopch : {} \t Loss : {:.6}".format(epoch, loss.item()))

# 8 定义测试方法
def test_model(model, device, test_loader):
    # 模型验证
    model.eval()
    # 统计正确率
    correct = 0
    # 测试损失
    test_loss = 0
    with torch.no_grad(): # 不用计算梯度，也不用进行反向传播
        for data, target in test_loader:
            # 部署到device上去
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            test_loss += F.cross_entropy(output, target).item()
            # 找到概率值最大的下标（索引）
            pred = output.max(1, keepdim=True)[1] # 值=output.max(1, keepdim=True)[0]，索引 = output.max(1, keepdim=True)[1]
            # pred = torch.max(output, dim=1)
            # 累计正确的值或数目
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test —— Average loss : {:.4f}, Accuracy : {:.3}\n".format(test_loss, 100 * correct / len(test_loader.dataset)))

# 9 调用 方法7/8
for epoch in range(1, EPOCHS + 1):
    # 调用训练模型
    train_model(model, DEVICE, train_loader, optimzer, epoch)

    # 调用测试模型
    test_model(model, DEVICE, test_loader)
