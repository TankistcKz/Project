import cv2, pytesseract
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
numberPlateCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
plat_detector =  cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
video = cv2.VideoCapture('images/video_2025-05-07_12-17-22.mp4')
num_list = []

if(video.isOpened()==False):
    print('Error Reading Video')


def carplate_extract(image):
    plate = plat_detector.detectMultiScale(image,scaleFactor=1.2,minNeighbors=5,minSize=(25,25))
    for (x,y,w,h) in plate:
        image = cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0),2)
        return image


while True:
    ret, frame = video.read()
    if ret == True:
        frame_gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gr = cv2.bilateralFilter(frame_gr, 20, 15, 15)
        frame_plate = carplate_extract(frame_gr)
        plt.imshow(frame_plate, cmap='gray')
        plt.axis('off')
        plt.show()  
        
        num = pytesseract.image_to_string(
            frame_gr,
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCEHKMOPTXY0123456789')
            
        if len(num) > 5:
            for w1 in 'ABCEHKMOPTXY':
                for n1 in '0123456789':
                    for n2 in '0123456789':
                        for n3 in '0123456789':
                            for w2 in 'ABCEHKMOPTXY':
                                for w3 in 'ABCEHKMOPTXY':
                                    a = f'{w1}{n1}{n2}{n3}{w2}{w3}'
                                    if a in num:
                                        num_list.append(a)
        
    else:
        break

print(set(num_list))
with open('num_file.txt', 'a') as file:
    for i in set(num_list):
        print(i, file=file)
    file.close()
                                
video.release()