import cv2
import numpy as np

# 图像列表
image_list = ["img1.jpg", "img2.jpg", "img3.jpg"]

# 遍历每个图像
for image_file in image_list:
    # 读取图像
    img = cv2.imread(image_file)

    # 获取图像尺寸
    height, width, channels = img.shape

    # 创建一个和原图像相同大小的数组，用于保存处理后的结果
    result = np.zeros((height, width, channels), dtype=np.uint8)

    # 遍历每个像素
    for y in range(height):
        for x in range(width):
            # 获取当前像素的 RGB 值
            b, g, r = img[y, x]

            # 计算新的像素值
            resultpixel = (int(r) + int(g) + int(b)) // 3

            # 设置新的像素值
            result[y, x] = [resultpixel, resultpixel, resultpixel]

    # 输出结果图像
    output_file = "output_" + image_file
    cv2.imwrite(output_file, result)
    cv2.imshow("output_image.jpg", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
