import cv2, pytesseract
import matplotlib.pyplot as plt 
import numpy as np
import easyocr
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'




def open_img(img_path):
    img = cv2.imread(img_path)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    return img


def carplate_extract(image, carplate_ml):
    carplate_rects = carplate_ml.detectMultiScale(image, scaleFactor=1.1, minNeighbors=0)

    for x, y, w, h in carplate_rects:
        img = image[y+15:y+h-10, x+15:x+w-20]
    return img


def enlarge_img(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized_img


def main():
    img_rgb = open_img(img_path='images/ru4384635.jpg')
    carplate_ml = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

    img = carplate_extract(img_rgb, carplate_ml)
    img = enlarge_img(img, 150)

    img_gr = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gr = cv2.bilateralFilter(img_gr, 20, 15, 15)


    plt.imshow(img_gr, cmap='gray')
    plt.axis('off')
    plt.show()

    print('Номер ТС: ', pytesseract.image_to_string(
        img_gr,
        config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABEKMHOPCTYX0123456789')
    )


if __name__ == '__main__':
    main()