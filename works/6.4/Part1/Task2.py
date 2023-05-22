import cv2
import os
import numpy as np
def sharpen_image(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def adjust_brightness(image, alpha, beta):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

def adjust_contrast(image, alpha):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha)
    return adjusted_image

def adjust_saturation(image, saturation):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:,:,1] = hsv_image[:,:,1] * saturation
    adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return adjusted_image

def process_image(image_path):
    image = cv2.imread(image_path)
    
    # 锐化处理
    sharpened_image = sharpen_image(image)
    
    # 调整亮度和对比度
    adjusted_image = adjust_brightness(sharpened_image, alpha=0.2, beta=5)
    adjusted_image = adjust_contrast(adjusted_image, alpha=5.0)
    
    # 调整饱和度
    adjusted_image = adjust_saturation(adjusted_image, saturation=1.5)
    
    # 保存处理后的图像
    output_path = 'output_pics2/' + os.path.basename(image_path)
    cv2.imwrite(output_path, adjusted_image)

# 处理文件夹中的所有图像
input_folder = 'pics'
output_folder = 'output_pics2'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        process_image(image_path)
