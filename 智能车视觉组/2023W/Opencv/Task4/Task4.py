import cv2

# 读取图像
img = cv2.imread('img.jpg')

# 分离BGR通道
b, g, r = cv2.split(img)

# 显示每个通道的图像
cv2.imshow('Blue Channel', b)
cv2.imshow('Green Channel', g)
cv2.imshow('Red Channel', r)

cv2.waitKey(0)
cv2.destroyAllWindows()
