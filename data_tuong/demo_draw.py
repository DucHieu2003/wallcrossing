import cv2

img = cv2.imread("192.168.1.4.jpg")

# Cac duong tuong mau (toa do tu trai sang phai theo mep tren tuong be tong)
demo_lines = [
    [(208, 133), (475, 158)],   # tuong be tong chinh giua-sau
    [(560, 168), (655, 150)],   # doan tuong/container ben phai xe ben
]

for p1, p2 in demo_lines:
    cv2.line(img, p1, p2, (0, 255, 0), 3)
    cv2.circle(img, p1, 6, (0, 0, 255), -1)
    cv2.circle(img, p2, 6, (0, 0, 255), -1)

cv2.imwrite("demo_preview.jpg", img)
print("Saved demo_preview.jpg")
