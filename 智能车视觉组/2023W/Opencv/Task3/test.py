import cv2
srcMat = cv2.imread("img.jpg")
ori = cv2.imread("img.jpg")
shallowMat = srcMat
deepMat = srcMat.copy()
height, width, _ = srcMat.shape
for threshold in range(0, 255, 5):
    print("当前阈值", threshold)
    for j in range(height):
        for i in range(width):
            average = int((ori[j, i][0] + ori[j, i][1] + ori[j, i][2]) / 3)
            if average > threshold:
                average = 255
            else:
                average = 0
            srcMat[j, i][0] = average
            srcMat[j, i][1] = average
            srcMat[j, i][2] = average
    cv2.imshow("ori ", ori)
    cv2.imshow("scrMat", srcMat)
    cv2.imshow("S (shallow copy)", shallowMat)
    cv2.imshow("D (deep copy)", deepMat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
