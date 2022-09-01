import random
import shutil
import matplotlib.pyplot as plt
import cv2
import os
import numpy
from tqdm import tqdm
def get_fileNum(path):
    print(len(os.listdir(path)))

def random_seg_image(img,seg_num,img_num):
    draw_switch=0
    m=seg_num
    h, w = img.shape[0], img.shape[1]
    grid_h = int(h * 1.0 / (m - 1) + 0.5)
    grid_w = int(w * 1.0 / (m - 1) + 0.5)
    h = grid_h * m
    w = grid_w * m

    image_re = cv2.resize(img, (h, w), cv2.INTER_LINEAR)
    print(image_re.dtype)
    gx, gy = numpy.meshgrid(numpy.linspace(0, w, m + 1), numpy.linspace(0, h, m + 1))
    gx = gx.astype(int)
    gy = gy.astype(int)
    divid_rimg = numpy.zeros([grid_h, grid_w, 3], numpy.uint8)
    if(draw_switch==1):
        boarder_img=numpy.zeros([h, w, 3], numpy.uint8)
        boarder_img=image_re[:,:,:]
    for j in range(8):
        r_centery = random.randint(int(grid_h / 2 + 0.5), int(h - grid_h / 2))  #the random picture centre point
        r_centerx = random.randint(int(grid_w / 2 + 0.5), int(w - grid_w / 2))
#       y and x in the image is reverse
        divid_rimg = image_re[r_centery - int(grid_h / 2):r_centery + int(grid_h / 2),
                     r_centerx - int(grid_w / 2):r_centerx + int(grid_w / 2), :]
        if (draw_switch == 1):
            boarder_img=cv2.rectangle(boarder_img, (r_centerx - int(grid_w / 2),r_centery - int(grid_h / 2)), (r_centerx + int(grid_w / 2),r_centery + int(grid_h / 2)), (255, 0, 255), 1)
        print(divid_rimg.shape)
        vector = []
        if (abs(w / 2 - r_centerx)<=5 and abs(h / 2 - r_centery)<=5):
            v_x=0
            v_y=0
        else:
            v_x=(w / 2 - r_centerx)
            v_y=-(h / 2 - r_centery)
        vector.append(v_x)
        vector.append(v_y)

        # -----------------------------normalization the vetor and use it as the filename--------------------
        # if(abs(h / 2 - r_centerx)>=abs(w / 2 - r_centery)):
        #     v_x=(h / 2 - r_centerx)/abs(h / 2 - r_centerx)
        #     vector.append(v_x)
        #     v_y=-(w / 2 - r_centery)/abs(h / 2 - r_centerx)
        #     v_y=round(v_y,2)
        #     vector.append(v_y)
        # else:
        #     v_x = (h / 2 - r_centerx) / abs(w / 2 - r_centery)
        #     v_x = round(v_x, 2)
        #     vector.append(v_x)
        #     v_y = -(w / 2 - r_centery) / abs(w / 2 - r_centery)
        #     vector.append(v_y)
        print(vector)
        print(divid_rimg.shape)
        divid_rimg=cv2.resize(divid_rimg,(74,74),cv2.INTER_LINEAR)
        save_path="../../new_Charlock_plus/"

        # cv2.imwrite("../../picture/"+str(vector[0]) + ".png", divid_rimg)
        # input()
        cv2.imwrite(save_path+str(vector[0])+"_"+str(vector[1])+"_"+str(img_num*8+j)+".png",divid_rimg)
        # img = cv2.arrowedLine(image_re, (int(r_centerx), int(r_centery)), (int(h / 2), int(w / 2)), (0, 0, 255), 2, 3,
                              # 0, 0.3)
        # print(r_centerx,r_centery)
        # cv2.circle(image_re, (r_centerx, r_centery), 2, (0, 140, 255), -1)
        # plt.imshow(image_re)
        # plt.show()
    if (draw_switch == 1):
        cv2.imwrite("../../ab_i/"+str(img_num)+".png",boarder_img)

def Uniform_size(path,size):
    pass

def cut_file(from_path, to_path):
    pathDir = os.listdir(from_path)

    picknumber = 200 # the number of the picture
    sample = random.sample(pathDir, picknumber)  # randomly pick the picture
    print(sample)
    for name in sample:
        shutil.move(from_path + name, to_path + name)
    return

# #check the picture size
# file_list=os.listdir("../../new_Charlock/")
#
# for file in file_list:
#     img = cv2.imread("../../new_Charlock/" + file)
#     print(img.shape)
#     input()
# ##########################################
# img=cv2.imread("../../new_Charlock/0.0_1.0_110.png")
# print(img.shape)

# random_seg_image(img,8)
def create_dataset():
    path="../../CharlocK_2/"
    file_list=os.listdir(path)
    img_num=0
    for file in file_list:
        print(file)
        img=cv2.imread(path+file)
        img = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)
        random_seg_image(img,5,img_num)
        img_num=img_num+1

def rlRandom_seg_image(img,seg_num,img_num):
    # if(img_num%2==0):
    #     return
    m = seg_num
    h, w = img.shape[0], img.shape[1]
    grid_h = int(h * 1.0 / (m - 1) + 0.5)
    grid_w = int(w * 1.0 / (m - 1) + 0.5)
    h = grid_h * m
    w = grid_w * m
    image_re = cv2.resize(img, (h, w), cv2.INTER_LINEAR)
    print(image_re.dtype)
    gx, gy = numpy.meshgrid(numpy.linspace(0, w, m + 1), numpy.linspace(0, h, m + 1))
    gx = gx.astype(int)
    gy = gy.astype(int)
    divid_rimg = numpy.zeros([grid_h, grid_w, 3], numpy.uint8)

    r_centery = random.randint(int(grid_h / 2 + 0.5), int(h - grid_h / 2))
    r_centerx = random.randint(int(grid_w / 2 + 0.5), int(w - grid_w / 2))
    while(abs(r_centerx-int(w/2))<=8 or abs(r_centery-int(h/2))<1 or abs(r_centerx-int(w/2))/abs(r_centery-int(h/2))<3):
        r_centery = random.randint(int(grid_h / 2 + 0.5), int(h - grid_h / 2))  # the random picture centre point
        r_centerx = random.randint(int(grid_w / 2 + 0.5), int(w - grid_w / 2))

    divid_rimg = image_re[r_centery - int(grid_w / 2):r_centery + int(grid_w / 2),
                     r_centerx - int(grid_h / 2):r_centerx + int(grid_h / 2), :]
    vector = []
    if (abs(w / 2 - r_centerx) <= 5 and abs(h / 2 - r_centery) <= 5):
        v_x = 0
        v_y = 0
    else:
        v_x = (w / 2 - r_centerx)
        v_y = -(h / 2 - r_centery)
    vector.append(v_x)
    vector.append(v_y)
    print(vector)
    divid_rimg = cv2.resize(divid_rimg, (74, 74), cv2.INTER_LINEAR)
    # save_path="../../nenen/"
    save_path="../../new_Charlock_plus/"
    cv2.imwrite(save_path + str(vector[0]) + "_" + str(vector[1]) + "_" + str(img_num) + "arl.png",
                divid_rimg)
def udRandom_seg_image(img,seg_num,img_num):
    # if(img_num%2==0):
    #     return
    m = seg_num
    h, w = img.shape[0], img.shape[1]
    grid_h = int(h * 1.0 / (m - 1) + 0.5)
    grid_w = int(w * 1.0 / (m - 1) + 0.5)
    h = grid_h * m
    w = grid_w * m
    image_re = cv2.resize(img, (h, w), cv2.INTER_LINEAR)
    print(image_re.dtype)
    gx, gy = numpy.meshgrid(numpy.linspace(0, w, m + 1), numpy.linspace(0, h, m + 1))
    gx = gx.astype(int)
    gy = gy.astype(int)
    divid_rimg = numpy.zeros([grid_h, grid_w, 3], numpy.uint8)

    r_centery = random.randint(int(grid_h / 2 + 0.5), int(h - grid_h / 2))
    r_centerx = random.randint(int(grid_w / 2 + 0.5), int(w - grid_w / 2))
    while(abs(r_centery-int(h/2))<=8 or abs(r_centerx-int(w/2))<1 or abs(r_centery-int(h/2))/abs(r_centerx-int(w/2))<3):
        r_centery = random.randint(int(grid_h / 2 + 0.5), int(h - grid_h / 2))  # the random picture centre point
        r_centerx = random.randint(int(grid_w / 2 + 0.5), int(w - grid_w / 2))

    divid_rimg = image_re[r_centery - int(grid_w / 2):r_centery + int(grid_w / 2),
                     r_centerx - int(grid_h / 2):r_centerx + int(grid_h / 2), :]
    vector = []
    if (abs(w / 2 - r_centerx) <= 5 and abs(h / 2 - r_centery) <= 5):
        v_x = 0
        v_y = 0
    else:
        v_x = (w / 2 - r_centerx)
        v_y = -(h / 2 - r_centery)
    vector.append(v_x)
    vector.append(v_y)
    print(vector)
    divid_rimg = cv2.resize(divid_rimg, (74, 74), cv2.INTER_LINEAR)
    # save_path = "../../nenen/"
    save_path = "../../new_Charlock_plus/"
    cv2.imwrite(save_path + str(vector[0]) + "_" + str(vector[1]) + "_" + str(img_num) + "aud.png",
                divid_rimg)
def centreRandom_seg_image(img,seg_num,img_num,times,centre_range=8):
    m = seg_num
    h, w = img.shape[0], img.shape[1]
    grid_h = int(h * 1.0 / (m - 1) + 0.5)
    grid_w = int(w * 1.0 / (m - 1) + 0.5)
    h = grid_h * m
    w = grid_w * m
    image_re = cv2.resize(img, (h, w), cv2.INTER_LINEAR)
    print(image_re.dtype)
    gx, gy = numpy.meshgrid(numpy.linspace(0, w, m + 1), numpy.linspace(0, h, m + 1))
    gx = gx.astype(int)
    gy = gy.astype(int)
    divid_rimg = numpy.zeros([grid_h, grid_w, 3], numpy.uint8)

    r_centery = random.randint(int(h / 2)-centre_range, int(h/2)+centre_range)  # the random picture centre point
    r_centerx = random.randint(int(w/2)-centre_range, int(w / 2)+centre_range)

    divid_rimg = image_re[r_centery - int(grid_w / 2):r_centery + int(grid_w / 2),
                     r_centerx - int(grid_h / 2):r_centerx + int(grid_h / 2), :]
    vector = []
    vector.append(0)
    vector.append(0)
    print(vector)
    divid_rimg = cv2.resize(divid_rimg, (74, 74), cv2.INTER_LINEAR)
    # save_path = "../../nenen/"
    # save_path = "../../new_Charlock_plus/"
    save_path = "../../Charlock_plus_allG/"
    cv2.imwrite(save_path + str(vector[0]) + "_" + str(vector[1]) + "_" + str(img_num)+"_"+str(times) + "ac.png",
                divid_rimg)

# for making the dataset average, generate data in some special position
def localRandom_seg_image(img,seg_num,img_num,bot,top,lef,rig):
    m=seg_num
    h, w = img.shape[0], img.shape[1]
    grid_h = int(h * 1.0 / (m - 1) + 0.5)
    grid_w = int(w * 1.0 / (m - 1) + 0.5)
    h = grid_h * m
    w = grid_w * m

    image_re = cv2.resize(img, (h, w), cv2.INTER_LINEAR)
    print(image_re.dtype)
    gx, gy = numpy.meshgrid(numpy.linspace(0, w, m + 1), numpy.linspace(0, h, m + 1))
    gx = gx.astype(int)
    gy = gy.astype(int)
    divid_rimg = numpy.zeros([grid_h, grid_w, 3], numpy.uint8)
    for j in range(1):
        # r_centery = random.randint(int(w / 2 - top), int(w / 2+top))   #the random picture centre point
        # r_centerx = random.randint(int(h / 2 +lef), int(h - grid_h / 2))

        r_centery = random.randint(int(w/2+lef),int(w-grid_w/2)) # the random picture centre point
        r_centerx = random.randint(int(h / 2 - top), int(h / 2 + top))


        divid_rimg = image_re[r_centery - int(grid_w / 2):r_centery + int(grid_w / 2),
                     r_centerx - int(grid_h / 2):r_centerx + int(grid_h / 2), :]
        # divid_rimg=image_re[0:184,0:184,:]
        vector = []
        if (abs(h / 2 - r_centerx)<=8 and abs(w / 2 - r_centery)<=8):
            v_x=0
            v_y=0
        else:
            v_x=(h / 2 - r_centerx)
            v_y=-(w / 2 - r_centery)
        vector.append(v_x)
        vector.append(v_y)

        # -----------------------------normalization the vetor and use it as the filename--------------------
        # if(abs(h / 2 - r_centerx)>=abs(w / 2 - r_centery)):
        #     v_x=(h / 2 - r_centerx)/abs(h / 2 - r_centerx)
        #     vector.append(v_x)
        #     v_y=-(w / 2 - r_centery)/abs(h / 2 - r_centerx)
        #     v_y=round(v_y,2)
        #     vector.append(v_y)
        # else:
        #     v_x = (h / 2 - r_centerx) / abs(w / 2 - r_centery)
        #     v_x = round(v_x, 2)
        #     vector.append(v_x)
        #     v_y = -(w / 2 - r_centery) / abs(w / 2 - r_centery)
        #     vector.append(v_y)
        print(vector)
        # cv2.imwrite("../../picture/"+str(vector[0]) + ".png", divid_rimg)
        cv2.imwrite("../../nenen/"+str(vector[0])+"_"+str(vector[1])+"_"+str(img_num*8+j)+".png",divid_rimg)
        # img = cv2.arrowedLine(image_re, (int(r_centerx), int(r_centery)), (int(h / 2), int(w / 2)), (0, 0, 255), 2, 3,
                              # 0, 0.3)
        # print(r_centerx,r_centery)
        # cv2.circle(image_re, (r_centerx, r_centery), 2, (0, 140, 255), -1)
        # plt.imshow(image_re)
        # plt.show()

def move_inconsonant(from_path, to_path):
    num_incon=0
    pathDir = os.listdir(from_path)
    for name in pathDir:
        img=cv2.imread(from_path+name)
        if(img.shape[0]!=74):
            shutil.move(from_path + name, to_path + name)
            num_incon=num_incon+1
    print(num_incon)
def add_data(times=0):
    path = "../../CharlocK_2/"
    file_list = os.listdir(path)
    img_num = 0
    for file in file_list:
        print(file)
        img = cv2.imread(path + file)
        img = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)
        centreRandom_seg_image(img, 5, img_num, times)
        # rlRandom_seg_image(img, 5, img_num)
        # udRandom_seg_image(img, 5, img_num)
        img_num = img_num + 1

def green_division(img):
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_green=numpy.array([30,40,40]) #the color threshold top value
    higher_green=numpy.array([100,255,255])  #the color threshol bottom value
    mask=cv2.inRange(img_hsv,lower_green,higher_green)
    re_img = cv2.bitwise_and(img, img, mask=mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    green_area=0
    for i in contours:
        green_area=green_area+cv2.contourArea(i)
    # print(green_area)
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    # cv2.imshow('jj',mask)
    # plt.imshow(mask)
    # plt.show()
    return green_area

def remove_nogreen(path):
    file_list = os.listdir(path)
    img_num=0
    for file in tqdm(file_list):

        img=cv2.imread(path+file)

        img = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)
        green_area=green_division(img)
        if(green_area>1000 and green_area<49000):
            img_num=img_num+1
            img = cv2.resize(img, (74, 74), cv2.INTER_LINEAR)
            cv2.imwrite("../../Charlock_plus_allG/"+file,img)
        # else:
        #     cv2.imshow('aa',img)
        #     cv2.waitKey(0)

# cut_file("../../Charlock_3.0_allG/","../../Charlock_3.0_allG_v/")
# remove_nogreen("../../new_Charlock_plus/")

# for i in range(11):
#     # create_dataset()
#     add_data(i+20)