import detect
import facedetect
import os
import shutil
import cv2
import datetime

if __name__ == '__main__':
    path = './runs/detect'
    opt = detect.parse_opt()
    detect.main(opt)
    detect_list = os.listdir('./runs/detect')
    img_path = path + '/' + detect_list[-1]
    print(img_path)
    imgs = os.listdir(img_path)
    for img in imgs:
        image = cv2.imread(img_path+'/'+img)
        cv2.putText(image, 'LondoBell', (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
        t = datetime.datetime.now().strftime('%F %T')
        cv2.putText(image, t, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
        cv2.imwrite('./output/'+img, image)
    facedetect.detect('./faces/', './output/')




