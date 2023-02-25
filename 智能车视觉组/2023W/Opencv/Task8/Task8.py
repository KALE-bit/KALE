import cv2
import numpy as np

# 读取图像
img = cv2.imread('8.png')

# 将图像从BGR转换为HSV颜色空间
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 提取红色像素
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv_img, lower_red, upper_red)

lower_red = np.array([170, 100, 100])
upper_red = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv_img, lower_red, upper_red)

# 将两个 mask 相加
mask = mask1 + mask2

# 对原始图像和 mask 进行按位与操作，提取红色部分
red_img = cv2.bitwise_and(img, img, mask=mask)

# 显示原始图像和提取出的红色部分
cv2.imshow('original', img)
cv2.imshow('red', red_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
