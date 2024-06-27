import re
import sys
import glob
import os
import glob
import cv2
import numpy as np
import pytesseract

carplate_haar_cascade = cv2.CascadeClassifier(
    '/Users/dheeraj/PycharmProjects/Number_Plate/haarcascade_russian_plate_number.xml')


def carplate_detect(image):
    carplate_overlay = image.copy()
    carplate_rects = carplate_haar_cascade.detectMultiScale(carplate_overlay, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in carplate_rects:
        cv2.rectangle(carplate_overlay, (x, y), (x + w, y + h), (255, 0, 0), 5)

    carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in carplate_rects:
        carplate_img = image[y + 15:y + h - 10, x + 15:x + w - 20]

        carplate_extract_img_gray = cv2.cvtColor(carplate_img, cv2.COLOR_RGB2GRAY)
        carplate_extract_img_gray_blur = cv2.medianBlur(carplate_extract_img_gray, 3)

    number_plate_f = pytesseract.image_to_string(carplate_extract_img_gray_blur,
                                                 config=f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    return number_plate_f


# quick sort
def partition(arr, low, high):
    i = (low - 1)
    pivot = arr[high]

    for j in range(low, high):
        if arr[j] < pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return (i + 1)


def quickSort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)

        quickSort(arr, low, pi - 1)
        quickSort(arr, pi + 1, high)

    return arr


def binarySearch(arr, l, r, x):
    if r >= l:
        mid = l + (r - l) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] > x:
            return binarySearch(arr, l, mid - 1, x)
        else:
            return binarySearch(arr, mid + 1, r, x)
    else:
        return -1


array = []
for img in glob.glob("images/*.jpeg"):
    img = cv2.imread(img)

    img2 = cv2.resize(img, (600, 600))
    cv2.imshow("Image of car ", img2)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    number_plate = carplate_detect(img)
    res2 = str("".join(re.split("[^a-zA-Z0-9]*", number_plate)))
    res2 = res2.upper()
    #print(res2)

    array.append(res2)

# Sorting
array = quickSort(array, 0, len(array) - 1)
print("\n\n")
print("The Vehicle numbers registered are:-")
for i in array:
    print(i)
print("\n\n")

# Searching

cam = cv2.VideoCapture('https://192.168.57.22:8080/video')
img_counter =0


while True:
    ret, frame  =cam.read()

    if not ret:
        print("Falied to get frame")
        break
    cv2.imshow('test',frame)

    k = cv2.waitKey(1)

    if k%256 == 27:
        print("Escape hit, closing the app")
        break
    elif k%256 == 32:
        img_name = 'opencv_frame.jpeg'
        cv2.imwrite(img_name,frame)
        print("captured")
        img_counter+=1

img = cv2.imread(img_name)

cam.release()
cv2.destroyAllWindows()

number_plate = carplate_detect(img)
res2 = str("".join(re.split("[^a-zA-Z0-9]*", number_plate)))

print("The car number to search is:- ", res2)

result = binarySearch(array, 0, len(array) - 1, res2)
if result != -1:
    print("\n\nThe Vehicle is allowed to visit.")
else:
    print("\n\nThe Vehicle is  not allowed to visit.")
