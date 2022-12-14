import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import os
import Simple_CNN
import Resnet
import original_LeNet


# train_data_path='../../new_Charlock_plus/'
train_data_path='../../Charlock_3.0_allG/'
vaid_data_path='../../Charlock_3.0_allG_v/'
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
class load_dataset(Dataset):
    def __init__(self,data_path):
        label_num=[0,0,0,0,0,0,0,0,0]
        self.images = []
        self.labels = []
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        for filename in tqdm(os.listdir(data_path)):
            image = Image.open(data_path + filename)
            # image.show()
            # print(filename)
            # print(image.size[0],image.size[1])
            # image = image.resize((224, 224))  # uniform image size
            # image = image.resize((32, 32))
            image = image.resize((74, 74))
            image = self.transform(image)  # transform to tensor
            self.images.append(image)
            # print(image.size())

            ##-----------------------set the label for every picture------------------------------------
            file_seg=filename.split('_')
            if float(file_seg[0])==0 and float(file_seg[1])==0: #at the centere
                self.labels.append(0)
                label_num[0]=label_num[0]+1
            elif float(file_seg[1])==0 or abs(float(file_seg[0])/float(file_seg[1]))>=3:
                if(float(file_seg[0])>0): #right
                    self.labels.append(1)
                    label_num[1] = label_num[1] + 1
                else:
                    self.labels.append(2)  #left
                    label_num[2] = label_num[2] + 1
            elif float(file_seg[0])==0 or abs(float(file_seg[1])/float(file_seg[0]))>=3:
                if (float(file_seg[1]) > 0):  #up
                    self.labels.append(3)
                    label_num[3] = label_num[3] + 1
                else:
                    self.labels.append(4)  #down
                    label_num[4] = label_num[4] + 1
            elif abs(float(file_seg[1])/float(file_seg[0]))<3 and abs(float(file_seg[0])/float(file_seg[1]))<3:
                if(float(file_seg[0])>0):
                    if(float(file_seg[1])>0):
                        self.labels.append(5) #right-up
                        label_num[5] = label_num[5] + 1
                    else:
                        self.labels.append(6)    #right-down
                        label_num[6] = label_num[6] + 1
                if (float(file_seg[0]) < 0):
                    if (float(file_seg[1]) > 0):    #left-up
                        self.labels.append(7)
                        label_num[7] = label_num[7] + 1
                    else:
                        self.labels.append(8)       #left-down
                        label_num[8] = label_num[8] + 1
        print(label_num)
            # print(self.labels)
            # input()


            # if float(filename.split('_')[0]) <=0.2 and float(filename.split('_')[0]) >=-0.2 and float(filename.split('_')[1]) ==1:  # mark label
            #     self.labels.append(0)
            # elif float(filename.split('_')[0]) <=0.2 and float(filename.split('_')[0]) >=-0.2 and float(filename.split('_')[1]) ==-1:
            #     self.labels.append(1)
            # elif float(filename.split('_')[0]) ==-1 and float(filename.split('_')[1]) <=0.2 and float(filename.split('_')[1]) >=-0.2:
            #     self.labels.append(2)
            # elif float(filename.split('_')[0]) ==1 and float(filename.split('_')[1]) <=0.2 and float(filename.split('_')[1]) >=-0.2:
            #     self.labels.append(3)
            # elif float(filename.split('_')[0]) >0.2 and float(filename.split('_')[1]) >0.2:
            #     self.labels.append(4)
            # elif float(filename.split('_')[0]) <-0.2 and float(filename.split('_')[1]) >0.2:
            #     self.labels.append(5)
            # elif float(filename.split('_')[0]) <-0.2 and float(filename.split('_')[1]) <-0.2:
            #     self.labels.append(6)
            # elif float(filename.split('_')[0]) >0.2 and float(filename.split('_')[1]) <-0.2:
            #     self.labels.append(7)
        self.labels = torch.LongTensor(self.labels)  # ???????????????Tensor??????

    def __getitem__(self, index):  # create iterator
        return self.images[index], self.labels[index]

    def __len__(self):  # len of the iterator
        # print("len is:")
        # print(type(self.images))
        len_oi=len(self.images)
        # print(len_oi)
        # input()
        # images = np.array(self.images)
        # len_image = images.shape[0]
        # print(len_image)
        return len_oi


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()

        self.conv1=nn.Conv2d(3,16,5)
        self.pool1=nn.MaxPool2d(2,2)

        self.conv2=nn.Conv2d(16,32,4)
        self.pool2=nn.MaxPool2d(2,2)
        # return self.pool2

        self.conv3=nn.Conv2d(32,64,3)
        self.pool3=nn.MaxPool2d(2,2)


        self.fc1 = nn.Linear(64*7*7,300)
        self.fc1_5=nn.Linear(300,120)
        # self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(120,9)


    def forward(self,x):

        x=F.relu(self.conv1(x))
        x=self.pool1(x)
        x=F.relu(self.conv2(x))
        x=self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x=x.view(-1,64*7*7)
        x=F.relu(self.fc1(x))
        x = F.relu(self.fc1_5(x))
        # x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

input1 = torch.rand([32,3,74,74])
# model = LeNet() # ???????????????
model=Simple_CNN.Layers_2()
# c1=model.conv1(input1)
# c1=model.pool1(c1)
# c1=model.conv2(c1)
# c1=model.pool2(c1)
# c1=model.conv3(c1)
# c1=model.pool3(c1)
#
# print(c1.shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = load_dataset(train_data_path)
print(len(train_set))
input()
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                               shuffle=True, num_workers=0)
print(len(train_loader))

val_set = load_dataset(vaid_data_path)
print(len(val_set))

val_loader = DataLoader(val_set, batch_size=90,
                        shuffle=False, num_workers=0)
print(len(val_loader))


# valid and train
train_size=int(len(train_set) * 0.8)
vaid_size = len(train_set)-train_size
train_dataset, vaid_dataset = torch.utils.data.random_split(train_set, [train_size, vaid_size])
train_dataset=torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                              shuffle=True, num_workers=0)
vaid_dataset=torch.utils.data.DataLoader(vaid_dataset, batch_size=200,
                                              shuffle=True, num_workers=0)
#######################################


val_data_iter = iter(val_loader) # use to iterate
val_image, val_label = val_data_iter.next()


# train_data_iter=iter(train_loader[:100])
train_image, train_label = next(iter(train_loader))
vaid_image, vaid_label = next(iter(vaid_dataset))
print("size of the train data: "+str(len(vaid_image)))

# net = Resnet.ResNet50()
net = Simple_CNN.Layers_2()
# net = original_LeNet.LeNet()
net.to(device)  # GPU
loss_function = nn.CrossEntropyLoss() # include softmax
optimizer = optim.Adam(net.parameters(), lr=0.0001) #Adam optimizer
log=open("log_tvt.txt",'w')

for epoch in range(300):  # loop over the dataset multiple times
    print("the %d time training"%(epoch))
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

            # zero the parameter gradients
        optimizer.zero_grad()

            # forward + backward + optimize ???????????????CPU
            # outputs = net(inputs)
            # loss = loss_function(outputs, labels)

            # GPU
        outputs = net(inputs.to(device))  # inputs sent to device
        loss = loss_function(outputs, labels.to(device))  # labels sent to device
        vaid_loss = loss_function(outputs, labels.to(device))
        loss.backward() # loss backforward
        optimizer.step() # step update parameter

            # print statistics
        running_loss += loss.item() # m after calculate put it into running_loss
        if step % 11 == 10:    # print every 500 mini-batches
            with torch.no_grad(): # ??????????????????????????????????????????????????????????????????????????????
                    # outputs = net(val_image)  # [batch, 10]
                outputs = net(val_image.to(device))  # ??????GPU???????????????test_image??????????????????device???
                outputs_t=net(train_image.to(device))
                outputs_v= net(vaid_image.to(device))
                # print(outputs)
                # input()
                predict_y = torch.max(outputs, dim=1)[1] #dim=1?????????dim=0???batch???[1]????????????????????????????????????
                    # accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
                    # eq???????????????????????????????????????1???????????????0 -> ????????????tensor?????????item???????????? CPU?????????
                predict_t_y = torch.max(outputs_t, dim=1)[1]
                predict_v_y = torch.max(outputs_v, dim=1)[1]


                accuracy = (predict_y==val_label.to(device)).sum().item() / val_label.size(0)
                accuracy_vaid= (predict_v_y==vaid_label.to(device)).sum().item() / vaid_label.size(0)
                accuracy_train = (predict_t_y==train_label.to(device)).sum().item() / train_label.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f  train_accuracy: %.3f  vaid_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy,accuracy_train,accuracy_vaid))
                log.write('['+str(epoch+1)+','+str(step+1)+'] train_loss:'+str(running_loss/500)+'  test_accuracy:'+str(accuracy)+'  train_accuracy:'+str(accuracy_train)+' vaid_accuracy:'+str(accuracy_vaid)+'\n')
                running_loss = 0.0

print('Finished Training')
log.write('Finished Training')
log.close()

save_path = './test_Lenet.pth'
torch.save(net.state_dict(), save_path)

# output = model(input1)
        # print("adasd")
        # data=torch.randn(2,3,74,74)
        # out=self.pool2(data)
        # print(out.shape)
# def forward(self, x):
#     # x???????????????????????????tensor
#     # ????????????
#     x = F.relu(self.conv1(x))  # input(3, 32, 32) output(16, 28, 28)
#     x = self.pool1(x)  # output(16, 14, 14)
#     x = F.relu(self.conv2(x))  # output(32, 10, 10)
#     x = self.pool2(x)  # output(32, 5, 5)
#     x = x.view(-1, 32 * 5 * 5)  # output(32*5*5)
#     # ????????????view????????????????????????????????????-1???batch??????????????????32x5x5?????????????????????
#     x = F.relu(self.fc1(x))  # output(120)
#     x = F.relu(self.fc2(x))  # output(84)
#     x = self.fc3(x)  # output(10)
#
#     # ??????????????????softmax?????? --- ???????????????????????????????????????????????????
#     return x

# def main():
#     net = LeNet()
#
#
# if __name__=="__main__":
#     main()

