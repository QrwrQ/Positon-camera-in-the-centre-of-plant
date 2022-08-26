import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

batch_size = 64

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])  # 设置transform
class load_dataset(Dataset):
    def __init__(self,data_path):
        self.images = []
        self.labels = []
        self.transform = transform
        for filename in tqdm(os.listdir(data_path)):
            image = Image.open(data_path + filename)
            print(image.size[0],image.size[1])
            image = image.resize((224, 224))  # uniform image size
            image = self.transform(image)  # transform to tensor
            self.images.append(image)
            print(image.size())
            if float(filename.split('_')[0]) <=0.2 and float(filename.split('_')[0]) >=-0.2 and float(filename.split('_')[1]) ==1:  # mark label
                self.labels.append(0)
            elif float(filename.split('_')[0]) <=0.2 and float(filename.split('_')[0]) >=-0.2 and float(filename.split('_')[1]) ==-1:
                self.labels.append(1)
            elif float(filename.split('_')[0]) ==-1 and float(filename.split('_')[1]) <=0.2 and float(filename.split('_')[1]) >=-0.2:
                self.labels.append(2)
            elif float(filename.split('_')[0]) ==1 and float(filename.split('_')[1]) <=0.2 and float(filename.split('_')[1]) >=-0.2:
                self.labels.append(3)
            elif float(filename.split('_')[0]) >0.2 and float(filename.split('_')[1]) >0.2:
                self.labels.append(4)
            elif float(filename.split('_')[0]) <-0.2 and float(filename.split('_')[1]) >0.2:
                self.labels.append(5)
            elif float(filename.split('_')[0]) <-0.2 and float(filename.split('_')[1]) <-0.2:
                self.labels.append(6)
            elif float(filename.split('_')[0]) >0.2 and float(filename.split('_')[1]) <-0.2:
                self.labels.append(7)
        self.labels = torch.LongTensor(self.labels)  # 标签转化位Tensor格式

    def __getitem__(self, index):  # create iterator
        return self.images[index], self.labels[index]

    def __len__(self):  # len of the iterator
        images = np.array(self.images)
        len = images.shape[0]
        return len

train_data = load_dataset('../../new_Charlock/')              #加载训练集
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

val_data = load_dataset('../../new_Charlock_v/')                  #加载验证集
val_loader = DataLoader(dataset=val_data, batch_size=64, shuffle=True)


class InceptionA(torch.nn.Module):                              #Inception layer
    def __init__(self,in_channels):
        super(InceptionA,self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels,16,kernel_size = 1)
        self.branch5x5_1 = torch.nn.Conv2d(in_channels,16,kernel_size = 1)
        self.branch5x5_2 = torch.nn.Conv2d(16,24,kernel_size = 5,padding = 2)
        self.branch3x3_1 = torch.nn.Conv2d(in_channels,16,kernel_size = 1)
        self.branch3x3_2 = torch.nn.Conv2d(16,24,kernel_size = 3,padding = 1)
        self.branch3x3_3 = torch.nn.Conv2d(24,24,kernel_size = 3,padding = 1)
        self.branch_pool = torch.nn.Conv2d(in_channels,24,kernel_size = 1)
    def forward(self,x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        branch_pool = F.avg_pool2d(x,kernel_size = 3,stride = 1,padding = 1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1,branch3x3,branch5x5,branch_pool]
        return torch.cat(outputs,dim = 1)
class Net(torch.nn.Module):                             #create the network
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,10,kernel_size = 5)
        self.conv2 = torch.nn.Conv2d(88,20,kernel_size = 5)
        self.incep1 = InceptionA(in_channels = 10)
        self.incep2 = InceptionA(in_channels = 20)
        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(247192,8)
    def forward(self,x):
        in_size = x.size(0)
        x = self.mp(F.relu((self.conv1(x))))
        x = self.incep1(x)
        x = self.mp(F.relu((self.conv2(x))))
        x = self.incep2(x)
        x = x.view(in_size,-1)
        x = self.fc(x)
        return x

model = Net()

criterion = torch.nn.CrossEntropyLoss()         #create loss function
optimizer = optim.SGD(model.parameters(),lr = 0.001)        #create option

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train(epoch):
    running_loss = 0.0  # 训练模型
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  # 梯度清零

        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, targets)  # 计算损失
        loss.backward()  # 逆向传播
        optimizer.step()  # 梯度递进
        running_loss += loss.item()

    print('train loss: %.3f' % (running_loss / batch_idx))


def val():  # 验证模型精度
    correct = 0
    total = 0
    with torch.no_grad():  # 不需要梯度，减少计算量
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))
    return correct / total


if __name__ == '__main__':
    accuracy_list = []
    epoch_list = []

    for epoch in range(10):
        train(epoch)
        acc = val()
        accuracy_list.append(acc)
        epoch_list.append(epoch)

    plt.plot(epoch_list, accuracy_list)
    plt.xlabel(epoch)
    plt.ylabel(accuracy_list)
    plt.show()
