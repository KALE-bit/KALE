import os
import glob
import xml.etree.ElementTree as ET

def rename_files(path, old_pattern, new_pattern):
    for file_name in os.listdir(path):
        if old_pattern in file_name:
            new_name = file_name.replace(old_pattern, new_pattern)
            os.rename(os.path.join(path, file_name), os.path.join(path, new_name))

# 用法示例
#rename_files('./dataset', '1', 'A')

def rename_allfiles(path):
    i = 1
    for file_name in os.listdir(path):
        file_extension = os.path.splitext(file_name)[1] # 获取文件扩展名
        new_name = 'data'+ str(i) + file_extension # 组合新的文件名
        os.rename(os.path.join(path, file_name), os.path.join(path, new_name))
        i += 1

# 用法示例
rename_allfiles('./pic')

def modify_filename(path):
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = os.path.splitext(os.path.basename(xml_file))[0]
        root.find('filename').text = filename
        tree.write(xml_file)


def main():
    path = './dataset'
    modify_filename(path)
    print('Successfully modified filenames in XML files.')


#main()
