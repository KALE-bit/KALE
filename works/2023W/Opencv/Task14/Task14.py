import cv2

def smooth_image(img_path):
    # 读取原始图像
    img = cv2.imread(img_path)

    # 双边滤波
    smoothed = cv2.bilateralFilter(img, 15, 100, 100)

    # 原始图像减去平滑后的图像，得到细节图像
    detail = cv2.subtract(img, smoothed)

    # 对细节图像进行缩小处理
    small = cv2.resize(detail, None, fx=0.25, fy=0.25)

    # 对缩小后的细节图像进行模糊处理
    small = cv2.medianBlur(small, 5)

    # 将缩小后的细节图像放大回原始大小
    detail = cv2.resize(small, (img.shape[1], img.shape[0]))

    # 平滑后的图像加上细节图像，得到最终结果
    result = cv2.add(smoothed, detail)

    return result

# 读取两张图片并进行磨皮处理
img1 = smooth_image('14-1.jpg')
img2 = smooth_image('14-2.jpg')

# 显示处理结果
cv2.imshow('14-1', img1)
cv2.imshow('14-2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
