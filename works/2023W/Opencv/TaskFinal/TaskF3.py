import cv2
import numpy as np

image = cv2.imread("3.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义红色范围
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)

lower_red = np.array([170, 50, 50])
upper_red = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv, lower_red, upper_red)

# 组合掩码
mask = cv2.bitwise_or(mask1, mask2)

# 形态学处理
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# 查找轮廓
contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制外接矩形
min_area = 1000 # 面积阈值
for contour in contours:
    area = cv2.contourArea(contour)
    if area < min_area:
        continue
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)

cv2.imshow("result", image)
cv2.waitKey(0)
cv2.destroyAllWindows
