import cv2
import numpy as np

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    # Baca frame dari kamera
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip horizontal

    # Konversi frame ke HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rentang warna merah dalam HSV
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    # Rentang warna biru dalam HSV
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    # Rentang warna hijau dalam HSV
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    # Masking untuk mendeteksi warna merah, biru, dan hijau
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(frame, frame, mask=mask_red | mask_blue | mask_green)

    # Menemukan kontur dan menggambar bounding box sesuai warna
    for mask, color, label in [(mask_red, (0, 0, 255), "Red"), (mask_blue, (255, 0, 0), "Blue"), (mask_green, (0, 255, 0), "Green")]:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Filter kontur kecil
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # Bounding box sesuai warna
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Menampilkan hasil
    cv2.imshow("Frame", frame)  # Tampilkan frame asli dengan bounding box
    cv2.imshow("Result", result)
    cv2.imshow("Mask Red", mask_red)  # Tampilkan mask merah
    cv2.imshow("Mask Blue", mask_blue)  # Tampilkan mask biru
    cv2.imshow("Mask Green", mask_green)  # Tampilkan mask hijau

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
