import numpy as np
import cv2

# 创建一个黑色的图像
img = np.zeros((512, 512, 3), np.uint8)

# 画圆
cv2.circle(img, (256, 256), 128, (0, 255, 0), -1)

# 画线段
cv2.line(img, (0, 0), (512, 512), (255, 0, 0), 4)

# 画矩形框
cv2.rectangle(img, (256, 0), (512, 256), (0, 0, 255), 2)

# 显示图像
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
