import cv2

# 读取原始图像
img = cv2.imread('1.png')

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 自适应阈值处理
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# 边缘检测
edges = cv2.Canny(thresh, 50, 150)

# 显示结果
cv2.imshow('Original', img)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
