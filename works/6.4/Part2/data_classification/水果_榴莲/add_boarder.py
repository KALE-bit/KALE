import os
import cv2
import numpy as np

def add_border(input_path, output_folder, border_width):
    # 读取图像
    image = cv2.imread(input_path)

    # 计算新图像的尺寸
    new_width = image.shape[1] + (2 * border_width)
    new_height = image.shape[0] + (2 * border_width)

    # 创建一个新的带有黄色边框的图像
    new_image = np.zeros((new_height, new_width, 3), np.uint8)
    new_image[:] = (0, 255, 255)  # 设置为黄色

    # 将原始图像放置在新图像的中心位置
    x_offset = border_width
    y_offset = border_width
    new_image[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image

    file_name = os.path.basename(input_path)
    output_path = os.path.join(output_folder, file_name)

    # 保存新图像
    cv2.imwrite(output_path, new_image)

def resize_image(input_path, output_folder, width, height):
    # 加载图像
    image = cv2.imread(input_path)
    
    # 获取原始图像的尺寸
    org_width, org_height = image.shape[1], image.shape[0]
    
    # 计算缩放比例
    scale = min(width / org_width, height / org_height)

    # 计算缩放后的新尺寸
    new_width = int(org_width * scale)
    new_height = int(org_height * scale)
    
    # 缩放图像
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
    file_name = os.path.basename(input_path)
    output_path = os.path.join(output_folder, file_name)
    
    # 保存新图像
    cv2.imwrite(output_path, resized_image)

def process_images_in_folder(input_folder, output_folder, border_width):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有图像文件
    for file_name in os.listdir(input_folder):
        # 确保文件是JPG图像
        if file_name.endswith('.jpg'):
            # 构建输入图像的路径
            input_path = os.path.join(input_folder, file_name)

            # 处理图像并保存结果到输出文件夹
            add_border(input_path, output_folder, border_width)
            
    for file_name in os.listdir(output_folder):
        # 确保文件是JPG图像
        if file_name.endswith('.jpg'):
            # 构建输入图像的路径
            input_path = os.path.join(output_folder, file_name)

            # 处理图像并保存结果到输出文件夹
            resize_image(input_path, output_folder, 454, 454)
            
# 调用函数并指定输入文件夹、输出文件夹和边框宽度
process_images_in_folder('input_folder', 'output_folder', 48)
