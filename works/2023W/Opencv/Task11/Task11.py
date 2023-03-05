import cv2

# 读取图像
img = cv2.imread("11.png", cv2.IMREAD_GRAYSCALE)

# 二值化
_, binary_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

# 连通域标记
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img)

# 筛选连通域
min_area = 2000  # 最小面积阈值
max_area = 10000  # 最大面积阈值
Paperclips_count = 0
for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    if area < min_area or area > max_area:
        continue
    Paperclips_count += 1
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, str(Paperclips_count), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# 显示图像并输出回形针个数

print("Paperclips count: ", Paperclips_count)
cv2.imshow("Paperclips", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

