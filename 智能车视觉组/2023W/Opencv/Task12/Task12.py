import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取一帧图像
    ret, frame = cap.read()

    # 中值滤波
    median = cv2.medianBlur(frame, 5)

    # 均值滤波
    mean = cv2.blur(frame, (5, 5))

    # 高斯滤波
    gauss = cv2.GaussianBlur(frame, (5, 5), 0)

    # 显示原始图像和滤波结果
    cv2.imshow("Original", frame)
    cv2.imshow("Median", median)
    cv2.imshow("Mean", mean)
    cv2.imshow("Gaussian", gauss)

    # 等待按下 q 键退出
    if cv2.waitKey(1) == ord("q"):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
