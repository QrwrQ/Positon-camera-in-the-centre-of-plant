import random

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


m=4
img=cv2.imread(r"../../new_Charlock/0.0_1.0_110.png")
print(img.shape)
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
