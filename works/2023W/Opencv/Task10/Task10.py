import cv2

# 读取图像
img = cv2.imread("10.png", cv2.IMREAD_GRAYSCALE)

# 二值化
_, binary_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

# 连通域标记
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img)

# 绘制外接矩形
for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# 显示图像并输出硬币个数

print("Coins count: ", num_labels - 1)
cv2.imshow("Coins", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


