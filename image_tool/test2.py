import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)
    # 把训练集读取，别分成一个一个批次的，shuffle可用于随机打乱；batch_size是一次处理36张图像
    # num_worker在windows下只能设置成0

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=0)
    # 验证集 一次拿出5000张1出来验证，不用打乱

    val_data_iter = iter(val_loader)  # 转换成可迭代的迭代器
    val_image, val_label = val_data_iter.next()
    # 转换成迭代器后，用next方法可以得到测试的图像和图像的标签值

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 这一部分用来看数据集
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize -> 反标准化处理
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0))) #（1,2,0）-> 代表高度、宽度 通道
    #     plt.show()
    #
    # # print labels
    # print(' '.join('%5s' % classes[val_label[j]] for j in range(4)))
    # imshow(torchvision.utils.make_grid(val_image))

    net = LeNet()
    net.to(device)  # 使用GPU时将网络分配到指定的device中，不使用GPU注释
    loss_function = nn.CrossEntropyLoss()  # 已经包含了softmax函数
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # Adam优化器

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            # 一般batch_size根据硬件设备来设置的，这个清楚历史梯度，不让梯度累计，可以让配置低的用户加快训练

            # forward + backward + optimize 、、、、、CPU
            # outputs = net(inputs)
            # loss = loss_function(outputs, labels)

            # GPU使用时添加，不使用时注释
            outputs = net(inputs.to(device))  # 将inputs分配到指定的device中
            loss = loss_function(outputs, labels.to(device))  # 将labels分配到指定的device中

            loss.backward()  # loss进行反向传播
            optimizer.step()  # step进行参数更新

            # print statistics
            running_loss += loss.item()  # m每次计算完后就加入到running_loss中
            if step % 500 == 499:  # print every 500 mini-batches
                with torch.no_grad():  # 在测试、预测过程中，这个函数可以优化内存，防止爆内存
                    # outputs = net(val_image)  # [batch, 10]
                    outputs = net(val_image.to(device))  # 使用GPU时用这行将test_image分配到指定的device中
                    predict_y = torch.max(outputs, dim=1)[1]  # dim=1，因为dim=0是batch；[1]是索引，最大值在哪个位置
                    # accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
                    # eq用来比较，如果预测正确返回1，错误返回0 -> 得到的是tensor，要用item转成数值 CPU时使用

                    accuracy = (predict_y == val_label.to(device)).sum().item() / val_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()

