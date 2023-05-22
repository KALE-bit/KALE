import cv2
import os
import numpy as np
def adjust_brightness(image, alpha, beta):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


def process_image_brighter(image_path):
    image = cv2.imread(image_path)
    

    adjusted_image = adjust_brightness(image, alpha=1.2, beta=5)

    output_path = 'output_pics3_brighter/' + os.path.basename(image_path)
    cv2.imwrite(output_path, adjusted_image)
def process_image_darker(image_path):
    image = cv2.imread(image_path)
    

    adjusted_image = adjust_brightness(image, alpha=0.8, beta=5)

    output_path = 'output_pics3_darker/' + os.path.basename(image_path)
    cv2.imwrite(output_path, adjusted_image)

# 处理文件夹中的所有图像
input_folder = 'pics'
output_folder_brighter = 'output_pics3_brighter'
output_folder_darker = 'output_pics3_darker'

# 创建输出文件夹
os.makedirs(output_folder_brighter, exist_ok=True)
os.makedirs(output_folder_darker, exist_ok=True)

# 遍历文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        process_image_brighter(image_path)
        process_image_darker(image_path)
