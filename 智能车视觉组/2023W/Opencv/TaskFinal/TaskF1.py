import cv2
import numpy as np

image = cv2.imread("1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 检测最大半径 50 最小半径 10 的圆
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=50)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 255), -1) # 绘制圆形

mask = np.zeros(gray.shape, dtype=np.uint8)
for (x, y, r) in circles:
    cv2.circle(mask, (x, y), r, 255, -1) # 在掩模上绘制圆形

yellow = np.zeros(image.shape, dtype=np.uint8)
yellow[:] = (0, 255, 255) # 创建黄色图像

# 将填充后的小洞画出
hole = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
hole = cv2.bitwise_or(hole, yellow, mask=mask)

# 将原图与填充后的小洞合并
result=cv2.add(image, hole)

cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
