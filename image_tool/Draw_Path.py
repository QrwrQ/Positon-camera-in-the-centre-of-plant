import torch
import cv2
import os
import random
import math
import matplotlib.pyplot as plt
# import CNN_2
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import original_LeNet
import division
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
            image = image.resize((74, 74))  # uniform image size
            image = self.transform(image)  # transform to tensor
            self.images.append(image)
            # print(image.size())

            ##-----------------------set the label for every picture------------------------------------
            file_seg=filename.split('_')
            if float(file_seg[0])==0 and float(file_seg[1])==0:
                self.labels.append(0)
                label_num[0]=label_num[0]+1
            elif float(file_seg[1])==0 or abs(float(file_seg[0])/float(file_seg[1]))>=3:
                if(float(file_seg[0])>0):
                    self.labels.append(1)
                    label_num[1] = label_num[1] + 1
                else:
                    self.labels.append(2)
                    label_num[2] = label_num[2] + 1
            elif float(file_seg[0])==0 or abs(float(file_seg[1])/float(file_seg[0]))>=3:
                if (float(file_seg[1]) > 0):
                    self.labels.append(3)
                    label_num[3] = label_num[3] + 1
                else:
                    self.labels.append(4)
                    label_num[4] = label_num[4] + 1
            elif abs(float(file_seg[1])/float(file_seg[0]))<3 or abs(float(file_seg[0])/float(file_seg[1]))<3:
                if(float(file_seg[0])>0):
                    if(float(file_seg[1])>0):
                        self.labels.append(5)
                        label_num[5] = label_num[5] + 1
                    else:
                        self.labels.append(6)
                        label_num[6] = label_num[6] + 1
                if (float(file_seg[0]) < 0):
                    if (float(file_seg[1]) > 0):
                        self.labels.append(7)
                        label_num[7] = label_num[7] + 1
                    else:
                        self.labels.append(8)
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
        self.labels = torch.LongTensor(self.labels)  # 标签转化位Tensor格式

    def __getitem__(self, index):  # create iterator
        return self.images[index], self.labels[index]

    def __len__(self):  # len of the iterator
        images = np.array(self.images)
        len = images.shape[0]
        return len

def predict_img(image):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "./Lenet.pth"
    # model = LeNet()
    model = original_LeNet.LeNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = transform(image)  # transform to tensor
    image = image.to(device)
    output = model(image)
    predict_value = torch.max(output, dim=1)[1]
    print("prediction1")
    print("prediction_value:"+str(predict_value))
    predict_value=np.array(predict_value)
    return predict_value[0]

def class_2_vector(class_num):
    vector=[]
    if class_num==0:
        vector=[0,0]
    if class_num==1:
        vector=[1,0]
    if class_num==2:
        vector=[-1,0]
    if class_num==3:
        vector=[0,1]
    if class_num==4:
        vector=[0,-1]
    if class_num==5:
        vector=[1,1]
    if class_num==6:
        vector=[1,-1]
    if class_num==7:
        vector=[-1,1]
    if class_num==8:
        vector=[-1,-1]
    return vector
def distance_change(start_position,end_position,centre_positin):
    distanc_start=math.sqrt(math.pow(start_position[0]-centre_positin[0],2)+math.pow(start_position[1]-centre_positin[1],2))
    distanc_end=math.sqrt(math.pow(end_position[0]-centre_positin[0],2)+math.pow(end_position[1]-centre_positin[1],2))
    distance_res = distanc_start-distanc_end
    return distance_res
def simulate_camera(image_path):
    show_switch=0
    image_cv=cv2.imread(image_path)

    image_cv=cv2.resize(image_cv, (224, 224), cv2.INTER_LINEAR)
    cv2.imwrite('temp.png', image_cv)
    capture_from_image=cv2.imread('temp.png')
    capture_from_image=cv2.resize(capture_from_image, (224, 224), cv2.INTER_LINEAR)
    m=8
    step_size=3
    if(show_switch==1):
        cv2.namedWindow('dvide_image', cv2.WINDOW_NORMAL)
        cv2.namedWindow('original_image', cv2.WINDOW_NORMAL)
    h, w = image_cv.shape[0], image_cv.shape[1]
    grid_h = int(h * 1.0 / (m - 1) + 0.5)
    grid_w = int(w * 1.0 / (m - 1) + 0.5)
    h = grid_h * m
    w = grid_w * m
    image_cv = cv2.resize(image_cv, (h, w), cv2.INTER_LINEAR)
    capture_from_image=cv2.resize(capture_from_image, (h, w), cv2.INTER_LINEAR)
    border_left=int(grid_h / 2 + 0.5)
    border_right=int(h - grid_h / 2)
    border_up=int(w - grid_w / 2)
    border_down=int(grid_w / 2 + 0.5)
    r_centery = random.randint(int(grid_w / 2 + 0.5), int(w - grid_w / 2))  # the random picture centre point
    r_centerx = random.randint(int(grid_h / 2 + 0.5), int(h - grid_h / 2))
    start_position=[r_centerx,r_centery]
    cv2.drawMarker(image_cv, (r_centerx, r_centery), (0,255,0), markerType=2, markerSize=5)
    camera_caught_img = capture_from_image[r_centery - int(grid_w / 2):r_centery + int(grid_w / 2),
                 r_centerx - int(grid_h / 2):r_centerx + int(grid_h / 2), :]
    print(r_centerx,r_centery)
    if(show_switch==1):
        cv2.imshow('dvide_image',camera_caught_img)
        cv2.imshow('original_image',image_cv)
        cv2.waitKey(10)
    camera_caught_img=cv2.resize(camera_caught_img, (32, 32), cv2.INTER_LINEAR)

    print(camera_caught_img.shape)
    camera_caught_img=cv2.cvtColor(camera_caught_img,cv2.COLOR_RGB2BGR)
    # print(camera_caught_img)
    camera_caught_img=Image.fromarray(camera_caught_img)
    # print(camera_caught_img)
    picture_class=predict_img(camera_caught_img)

    move_vector=class_2_vector(picture_class)
    move_centery=r_centery-(move_vector[1]*5*3)
    move_centerx=r_centerx+move_vector[0]*5*3
    if(move_centerx>border_right or move_centerx<border_left or move_centery>border_up or move_centery<border_down):
        move_centery = r_centery + (move_vector[1] * 5 * 1)
        move_centerx = r_centerx - move_vector[0] * 5 * 1

    # print(move_centerx,move_centery)
    cv2.arrowedLine(image_cv,pt1=(r_centerx,r_centery), pt2=(move_centerx, move_centery), color=(255, 255, 0), thickness=1,
                    line_type=cv2.LINE_8, shift=0, tipLength=0.05)
    # cv2.arrowedLine(image_cv, pt1=(0, 0), pt2=(100, 100), color=(255, 255, 0),
    #                 thickness=1,
    #                 line_type=cv2.LINE_8, shift=0, tipLength=0.05)
    walk_time=0

    while(picture_class!=0 and walk_time<8):
        r_centerx=move_centerx
        r_centery=move_centery

        camera_caught_img = capture_from_image[r_centery - int(grid_w / 2):r_centery + int(grid_w / 2),
                            r_centerx - int(grid_h / 2):r_centerx + int(grid_h / 2), :]
        if(show_switch==1):
            cv2.imshow('dvide_image', camera_caught_img)
            cv2.imshow('original_image',image_cv)
            cv2.waitKey(10)
        camera_caught_img = cv2.resize(camera_caught_img, (32, 32), cv2.INTER_LINEAR)
        green_area = division.green_division(camera_caught_img)
        if(green_area>20 and green_area<1000):
            camera_caught_img = cv2.cvtColor(camera_caught_img, cv2.COLOR_RGB2BGR)
            camera_caught_img = Image.fromarray(camera_caught_img)
            picture_class = predict_img(camera_caught_img)
            move_vector = class_2_vector(picture_class)
        move_centery = r_centery - (move_vector[1] * 5 *step_size)
        move_centerx = r_centerx + move_vector[0] * 5 *step_size
        if (move_centerx > border_right or move_centerx < border_left or move_centery > border_up or move_centery < border_down):
            move_centery = r_centery + (move_vector[1] * 5 * step_size)
            move_centerx = r_centerx - move_vector[0] * 5 * step_size
        cv2.arrowedLine(image_cv, pt1=(r_centerx, r_centery), pt2=(move_centerx, move_centery), color=(255, 255, 0),
                        thickness=1,
                        line_type=cv2.LINE_8, shift=0, tipLength=0.05)
        walk_time=walk_time+1
    end_position=[move_centerx,move_centery]

    cv2.drawMarker(image_cv, (move_centerx, move_centery), (255, 0, 0), markerType=3, markerSize=5)
    distance_gap=distance_change(start_position,end_position,[int(w/2),int(h/2)])
    print("walk time"+str(walk_time))
    print(image_path.split('/')[3])
    cv2.imwrite("../../camera_path/"+image_path.split('/')[3],image_cv)
    image_cv = cv2.resize(image_cv, (224*3, 224*3), cv2.INTER_LINEAR)
    print(image_cv.shape)
    # cv2.imshow("track",image_cv)
    os.remove('temp.png')
    cv2.destroyAllWindows()
    return distance_gap

def distance_show(path):
    file_list = os.listdir(path)
    all_distance=[]
    for file in file_list:
        all_distance.append(simulate_camera(path+file))
    plt.plot(all_distance, label='distance')
    plt.xlabel('number')
    plt.ylabel('distance')
    plt.legend(loc='best')
    plt.show()
if __name__=="__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model_path="./Lenet.pth"
    # image_path="../../picture/"
    # model=LeNet()
    # model.load_state_dict(torch.load(model_path))
    # model.eval()
    # # net.to(device)
    #
    # pathDir = os.listdir(image_path)
    # sample_path = random.sample(pathDir, 1)
    # # print(sample_path[0])
    # img = cv2.imread(image_path+sample_path[0])
    # print("cv2image:")
    # print(img.shape)
    # print(img)
    # print("after transfer to PIL")
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img=Image.fromarray(img)
    # print(np.array(img))
    # # img = cv2.resize(img, (224,224), cv2.INTER_LINEAR)
    # # prediction=net(img)
    # image = Image.open(image_path+sample_path[0])
    # class_n=predict_img(image)
    # vector=class_2_vector(class_n)
    # print("class_n")
    # print(class_n)
    # print(vector)
    # print("PILimage:")
    # # image=np.array(image)
    # # image=np.array(image)
    # # image=image.resize((74,74))
    # # print(image)
    # # image = image.resize((74, 74))  # uniform image size
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # image = transform(image)  # transform to tensor
    # # print(image)
    # image=image.to(device)
    # output=model(image)
    # predict_value = torch.max(output, dim=1)[1]
    # print("prediction")
    # print(predict_value)
    # predict_value=predict_value.numpy()
    # print(predict_value[0])
    # vector=[]
    # simulate_camera("89.png")
    distance_show('../../CharlocK_2/')
