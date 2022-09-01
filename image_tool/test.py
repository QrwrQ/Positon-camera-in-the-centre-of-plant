import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy
def random_seg_image(img,seg_num):
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
    for j in range(8):
        r_centery = random.randint(int(grid_w / 2 + 0.5), int(w - grid_w / 2))
        r_centerx = random.randint(int(grid_h / 2 + 0.5), int(h - grid_h / 2))

        divid_rimg = image_re[r_centery - int(grid_w / 2):r_centery + int(grid_w / 2),
                     r_centerx - int(grid_h / 2):r_centerx + int(grid_h / 2), :]
        # divid_rimg=image_re[0:184,0:184,:]
        vector = []
        vector.append(h / 2 - r_centerx)
        vector.append(-(w / 2 - r_centery))
        print(vector)
        # cv2.imwrite("../../picture/"+str(vector[0]) + ".png", divid_rimg)
        cv2.imwrite("../../picture/"+str(vector[0])+"_"+str(vector[1])+".png",divid_rimg)
        img = cv2.arrowedLine(image_re, (int(r_centerx), int(r_centery)), (int(h / 2), int(w / 2)), (0, 0, 255), 2, 3,
                              0, 0.3)
        # print(r_centerx,r_centery)
        cv2.circle(image_re, (r_centerx, r_centery), 2, (0, 140, 255), -1)
        plt.imshow(image_re)
        plt.show()
def draw_range(seg_num,img_num):
    img=cv2.imread('../89.png')
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
    boarder_img = cv2.rectangle(image_re, (int(grid_w / 2), int(grid_h / 2)),
                                (w - int(grid_w / 2),h - int(grid_h / 2)), (255, 0, 0), 3)
    cv2.imwrite("boader.png",boarder_img)
    cv2.imshow("11",boarder_img)
    cv2.waitKey(0)


def split_imag():
# 读取图片
    im = Image.open('../PP.png')

# 宽高各除 3，获取裁剪后的单张图片大小
    width = im.size[0]//3
    height = im.size[1]//3

# 裁剪图片的左上角坐标
    start_x = 0
    start_y = 0

# 用于给图片命名
    im_name = 1

# 循环裁剪图片
    for i in range(3):
        for j in range(3):
        # 裁剪图片并保存
            crop = im.crop((start_x, start_y, start_x+width, start_y+height))
        # 判断文件夹是否存在
        #     if not os.path.exists('imgs'):
                # os.mkdir('imgs')
            # crop.save('imgs/' + str(im_name) + '.jpg')
            plt.subplot(3,3,i*3+j+1)
            plt.imshow(crop)
            plt.axis('off')

        # 将左上角坐标的 x 轴向右移动
            start_x += width
            im_name += 1

    # 当第一行裁剪完后 x 继续从 0 开始裁剪
        start_x = 0
    # 裁剪第二行
        start_y += height
    plt.show()
draw_range(6,1)
# m=4
# img=cv2.imread(r"../../new_Charlock/0.0_1.0_110.png")
# print(img.shape)
# random_seg_image(img,8)
# print(img.shape)
# h,w=img.shape[0],img.shape[1]
# grid_h=int(h*1.0/(m-1)+0.5)
# grid_w=int(w*1.0/(m-1)+0.5)
# h=grid_h*m
# w=grid_w*m
#
# image_re=cv2.resize(img,(h,w),cv2.INTER_LINEAR)
# print(image_re.dtype)
# gx,gy=numpy.meshgrid(numpy.linspace(0,w,m+1),numpy.linspace(0,h,m+1))
# gx=gx.astype(int)
# gy=gy.astype(int)
# print(int(grid_w/2))
# divid_rimg=numpy.zeros([grid_h,grid_w,3],numpy.uint8)
# for j in range(8):
#     r_centery=random.randint(int(grid_w/2+0.5),int(w-grid_w/2))
#     r_centerx=random.randint(int(grid_h/2+0.5),int(h-grid_h/2))
#
#     divid_rimg=image_re[r_centery-int(grid_w/2):r_centery+int(grid_w/2),r_centerx-int(grid_h/2):r_centerx+int(grid_h/2),:]
# # divid_rimg=image_re[0:184,0:184,:]
#     vector=[]
#     vector.append(h/2-r_centerx)
#     vector.append(-(w/2-r_centery))
#     print(vector)
#     img = cv2.arrowedLine(image_re, (int(r_centerx),int(r_centery)), (int(h/2),int(w/2)), (0,0,255),2,3,0,0.3)
# # print(r_centerx,r_centery)
#     cv2.circle(image_re, (r_centerx, r_centery), 2, (0, 140, 255), -1)
#     plt.imshow(image_re)
#     plt.show()





# divid_image=numpy.zeros([m,m,grid_h,grid_w,3],numpy.uint8)
# for i in range(m):
#     for j in range(m):
#         divid_image[i,j,...]=image_re[gy[i][j]:gy[i+1][j+1],gx[i][j]:gx[i+1][j+1],:]
# re_img=numpy.zeros([grid_h,grid_w,3],numpy.uint8)
# re_img=divid_image[0,0,:]
# cv2.imwrite("re1.png",re_img)
# plt.imshow(re_img)
# plt.show()

# for i in range(m):
#     for j in range(m):
#         plt.subplot(m,m,i*m+j+1)
#         plt.imshow(divid_image[i,j,:])
#         plt.axis('off')
# plt.show()
# print(divid_image.dtype)
# plt.imshow(image_re)
# plt.show()
# print(img.shape)
