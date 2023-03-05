import cv2
import numpy as np

# 定义 Gamma 矫正函数
def gamma_correction(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = (np.array([((i / 255.0) ** inv_gamma) * 255
                       for i in np.arange(0, 256)]).astype("uint8"))
    return cv2.LUT(image, table)

# 设置 Gamma 值
gamma = 2

# 定义图像列表
image_list = ['7-1.png','7-2.jpg']

# 循环读取每个图像并进行 Gamma 矫正
for image_file in image_list:
    # 读取图像
    image = cv2.imread(image_file)

    # 进行 Gamma 校正
    corrected_image = gamma_correction(image, gamma)
    
    # 显示原图和 Gamma 校正后的图像
    cv2.imshow("Original", image)
    cv2.imshow("Corrected", corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
