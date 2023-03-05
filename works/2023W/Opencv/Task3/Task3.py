import cv2
import numpy as np

# 读取图像
srcMat = cv2.imread("img.jpg")
O = cv2.imread("img.jpg")

# 深复制
deepMat = srcMat.copy()

# 浅复制
shallowMat = srcMat

# 设置阈值
threshold = 150

# 遍历每个像素
for y in range(srcMat.shape[0]):
    for x in range(srcMat.shape[1]):
        # 获取当前像素的 RGB 值
        b, g, r = srcMat[y, x]

        # 计算新的像素值
        average = (int(r) + int(g) + int(b)) // 3
        if average > threshold:
            average = 255
        else:
            average = 0

        # 将新的像素值设置到 srcMat 中
        srcMat[y, x] = [average, average, average]

# 显示结果
cv2.imshow("origin", O)
cv2.imshow("srcMat", srcMat)
cv2.imshow("deepMat", deepMat)
cv2.imshow("shallowMat", shallowMat)
cv2.waitKey(0)
cv2.destroyAllWindows()
