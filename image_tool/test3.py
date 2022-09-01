import cv2
import os
def show():
    path="../../CharlocK_2/"
    file_list=os.listdir(path)
    cv2.namedWindow('dvide_image', cv2.WINDOW_NORMAL)
    img = cv2.imread('../PP.png')
    cv2.imwrite('temp.png',img)
    img_2=cv2.imread('temp.png')
    cv2.arrowedLine(img, pt1=(0, 0), pt2=(50, 50), color=(255, 255, 0),
                    thickness=2,
                    line_type=cv2.LINE_8, shift=0, tipLength=0.05)
    cv2.imshow('dvide_image', img)
    cv2.waitKey(0)
    os.remove('temp.png')
    # cv2.waitKey(10000)
    # for name in file_list:
    #     img=cv2.imread(path+name)
    #     # img=cv2.resize(img,(224,224),cv2.INTER_LINEAR)
    #     # print(img)
    #     cv2.imshow('dvide_image',img)
    #     cv2.waitKey(1000)
if __name__=="__main__":
    show()