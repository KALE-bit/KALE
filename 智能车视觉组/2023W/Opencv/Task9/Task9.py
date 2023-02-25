import cv2
import numpy as np

# 读取图像
img = cv2.imread('9.png', cv2.IMREAD_GRAYSCALE)

# 定义不同形状的算子

# 矩形
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 椭圆形
ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 交叉形
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

# 腐蚀操作
erosion_rect = cv2.erode(img, rect_kernel)
erosion_ellipse = cv2.erode(img, ellipse_kernel)
erosion_cross = cv2.erode(img, cross_kernel)

# 膨胀操作
dilation_rect = cv2.dilate(img, rect_kernel)
dilation_ellipse = cv2.dilate(img, ellipse_kernel)
dilation_cross = cv2.dilate(img, cross_kernel)

# 开运算
opening_rect = cv2.morphologyEx(img, cv2.MORPH_OPEN, rect_kernel)
opening_ellipse = cv2.morphologyEx(img, cv2.MORPH_OPEN, ellipse_kernel)
opening_cross = cv2.morphologyEx(img, cv2.MORPH_OPEN, cross_kernel)

# 闭运算
closing_rect = cv2.morphologyEx(img, cv2.MORPH_CLOSE, rect_kernel)
closing_ellipse = cv2.morphologyEx(img, cv2.MORPH_CLOSE, ellipse_kernel)
closing_cross = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cross_kernel)

# 显示处理结果
cv2.imshow('Original Image', img)
cv2.imshow('Erosion (Rect)', erosion_rect)
cv2.imshow('Erosion (Ellipse)', erosion_ellipse)
cv2.imshow('Erosion (Cross)', erosion_cross)
cv2.imshow('Dilation (Rect)', dilation_rect)
cv2.imshow('Dilation (Ellipse)', dilation_ellipse)
cv2.imshow('Dilation (Cross)', dilation_cross)
cv2.imshow('Opening (Rect)', opening_rect)
cv2.imshow('Opening (Ellipse)', opening_ellipse)
cv2.imshow('Opening (Cross)', opening_cross)
cv2.imshow('Closing (Rect)', closing_rect)
cv2.imshow('Closing (Ellipse)', closing_ellipse)
cv2.imshow('Closing (Cross)', closing_cross)

# 等待按键关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
