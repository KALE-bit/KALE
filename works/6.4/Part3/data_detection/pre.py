import os
import random
from PIL import Image

# 设置图像数据集的路径和保存预处理后图像的路径
dataset_path = './Images'
output_path = './outputImages'

# 定义裁剪和旋转的参数范围
crop_width = 240
crop_height = 180
rotation_angle_range = (-10, 10)

# 遍历图像数据集中的每个图像文件
for filename in os.listdir(dataset_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 拼接图像文件的完整路径
        image_path = os.path.join(dataset_path, filename)

        # 打开图像文件
        image = Image.open(image_path)

        # 随机裁剪
        image_width, image_height = image.size
        x = random.randint(0, image_width - crop_width)
        y = random.randint(0, image_height - crop_height)
        cropped_image = image.crop((x, y, x + crop_width, y + crop_height))

        # 保存裁剪后的图像
        cropped_output_filename = os.path.join(output_path, 'cropped_' + filename)
        cropped_image.save(cropped_output_filename)

        # 镜像
        mirrored_image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 保存镜像后的图像
        mirrored_output_filename = os.path.join(output_path, 'mirrored_' + filename)
        mirrored_image.save(mirrored_output_filename)

        # 随机旋转
        angle = random.randint(rotation_angle_range[0], rotation_angle_range[1])
        rotated_image = cropped_image.rotate(angle)

        # 保存旋转后的图像
        rotated_output_filename = os.path.join(output_path, 'rotated_' + filename)
        rotated_image.save(rotated_output_filename)
