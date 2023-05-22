from PIL import Image
import os

# 文件夹路径
folder_path = 'pics'

# 获取文件夹内所有图像文件
image_files = [f for f in os.listdir(folder_path) if not f.startswith(('_')) and f.endswith(('.jpg', '.jpeg', '.png'))]

# 循环处理每个图像文件
for file_name in image_files:

    image_path = os.path.join(folder_path, file_name)
    
    # 打开图像文件
    image = Image.open(image_path)
    
    # 创建一个新的带有蓝色背景的图像
    new_image = Image.new('RGBA', image.size, (0, 0, 255, 255))
    new_image.paste(image.convert('RGBA'), (0, 0), image.convert('RGBA'))
    
    # 旋转图像
    rotated_image = new_image.rotate(5, expand=True)
    
    # 保存处理后的图像
    new_file_name = '_' + file_name
    new_file_path = os.path.join(folder_path, new_file_name)
    rotated_image.save(new_file_path, format='PNG')
    
rotated_image_files = [f for f in os.listdir(folder_path) if f.startswith(('_')) and f.endswith(('.jpg', '.jpeg', '.png'))]    

for file_name in rotated_image_files:
    
    image_path = os.path.join(folder_path, file_name)
    
    # 打开图像文件
    image = Image.open(image_path)
    
    # 创建一个新的带有蓝色背景的图像
    new_image = Image.new('RGBA', image.size, (0, 0, 255, 255))
    new_image.paste(image.convert('RGBA'), (0, 0), image.convert('RGBA'))
    
    # 保存处理后的图像
    new_file_name = 'output' + file_name
    new_file_path = os.path.join(folder_path, new_file_name)
    new_image.save(new_file_path, format='PNG')
    os.remove(image_path)
