import cv2
import numpy as np
import os
import shutil



def change_brightness_contrast(image, alpha=1.0, beta=0):
    """
    调整图像的亮度和对比度。

    参数:
    image: 输入的图片 (NumPy数组)
    alpha: 对比度调整系数 (1.0 为原始对比度)
    beta: 亮度调整值 (0 为原始亮度)

    返回:
    调整后的图片 (NumPy数组)
    """
    # 调整对比度和亮度
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


img_list = os.listdir("test_ST_img")

if not os.path.exists("ap=1beta50/" + "ap=1beta50_img") and not os.path.exists("ap=0.5beta70/" + "ap=0.5beta70_msk"):
        os.mkdir("ap=1beta50/" + "ap=1beta50_img")
        os.mkdir("ap=1beta50/" + "ap=1beta50_msk")


img_path = "ap=1beta50/" + "ap=1beta50_img/"
mask_path = "ap=1beta50/" + "ap=1beta50_msk/"

for nm in img_list:
    images = cv2.imread("test_ST_img/" + nm, -1)
    light = change_brightness_contrast(images, alpha=1, beta=-70)
    cv2.imwrite(img_path + nm.replace("tif", "png"), light)
    
    
