import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)

# 判断摄像头是否成功打开
if cap.isOpened():
    print("已打开摄像头")
else:
    print("无法打开摄像头")
    exit()
# 设置窗口标题
cv2.namedWindow("Video")

# 计时器
start_time = cv2.getTickCount()
frame_count = 0

# 循环读取图片并显示
while True:
    # 读取图片
    ret, frame = cap.read()

    # 判断图片是否读取成功
    if not ret:
        print("无法读取图片")
        break

    # 显示每帧图片
    cv2.imshow("Video", frame)

    # 统计帧率
    frame_count += 1
    if (cv2.getTickCount() - start_time) / cv2.getTickFrequency() >= 1:
        fps = frame_count / ((cv2.getTickCount() - start_time) / cv2.getTickFrequency())
        print("FPS: {:.2f}".format(fps))
        frame_count = 0
        start_time = cv2.getTickCount()

    # 等待按下 q 键退出
    if cv2.waitKey(1) == ord("q"):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
