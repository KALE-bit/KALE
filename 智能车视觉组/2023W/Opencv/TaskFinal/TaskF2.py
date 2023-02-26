import cv2
import numpy as np

image = cv2.imread("2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 平滑处理
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 形态学处理
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
opened = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)

# 边缘检测
edges = cv2.Canny(opened, 50, 150, apertureSize=3)

# 最小矩形检测
L = 75 # 最小矩形边长阈值
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    rect = cv2.minAreaRect(contour) # 获取最小矩形
    (x, y), (w, h), angle = rect
    if w >= L and h >= L: # 如果矩形的边长不小于L，则绘制矩形
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 0, 255), 3)

cv2.imshow("result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
