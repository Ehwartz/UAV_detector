# encoding:utf-8
import requests
from base64 import b64encode
import os
import cv2
from time import sleep


# 获取百度人脸识别接口
def getToken():
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=19E3Ez0TsG0XBwn9hVD2nqlW&client_secret=2XZOQ3eiOT1svCSIGvBdf9Yjs8qfXx2n'
    response = requests.get(host)
    content = response.json()
    content = content['access_token']
    return content


# 人脸的类，需要包含的东西
class Face(object):
    def __init__(self, img, location, gender, match=0, name=None):
        self.img = img
        self.location = location
        self.gender = gender
        self.match = match  # 人脸是否匹配，默认为0
        self.name = name


# 获取人脸识别数据

def getData(imgpath):
    requestUrl = 'https://aip.baidubce.com/rest/2.0/face/v3/detect'
    token = getToken()
    params = {'access_token': token}
    f = open(imgpath, 'rb')
    temp = f.read()
    image = b64encode(temp)
    data = {
        'image': image,
        'image_type': 'BASE64',
        'max_face_num': '5',
        'face_field': 'gender'
    }
    response = requests.post(requestUrl, params=params, data=data)
    # print('响应结果：', response)
    content = response.json()
    # print('解析结果：', content)
    faces_detect = []
    face_num = content['result']['face_num']
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    for i in range(face_num):
        location = content['result']['face_list'][i]['location']
        leftTopX = int(location['left'])
        leftTopY = int(location['top'])
        rightBottomX = int(leftTopX + int(location['width']))
        rightBottomY = int(leftTopY + int(location['height']))
        img_cut = img[leftTopY:rightBottomY, leftTopX:rightBottomX]
        gender = content['result']['face_list'][i]['gender']['type']
        faces_detect.append(Face(img_cut, location, gender))
    return faces_detect


def draw(imgpath, faces_detect):
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    for face in faces_detect:
        leftTopX = int(face.location['left'])
        leftTopY = int(face.location['top'])
        rightBottomX = int(leftTopX + int(face.location['width']))
        rightBottomY = int(leftTopY + int(face.location['height']))
        cv2.rectangle(img, (leftTopX, leftTopY), (rightBottomX, rightBottomY), (0, 255, 0), 2)
        cv2.putText(img, face.gender, (leftTopX, leftTopY + 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
        if (face.match == 1):
            cv2.putText(img, face.name, (leftTopX, leftTopY - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))

    cv2.imwrite(os.path.split(imgpath)[0] + os.path.split(imgpath)[1].split('.')[0] + 'detect.jpg', img)
    print('oneimage over')


def draw_fast(imgpath, faces_detect):
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    for face in faces_detect:
        leftTopX = int(face.location['left'])
        leftTopY = int(face.location['top'])
        rightBottomX = int(leftTopX + int(face.location['width']))
        rightBottomY = int(leftTopY + int(face.location['height']))
        cv2.rectangle(img, (leftTopX, leftTopY), (rightBottomX, rightBottomY), (0, 255, 0), 2)
        cv2.putText(img, face.gender, (leftTopX, leftTopY + 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))


def search(s, name):
    path = []
    file = [x for x in os.listdir(s) if os.path.isfile(os.path.join(s, x)) and name in os.path.split(x)[1]]
    for x in file:
        path.append(s + os.path.normpath(x))
    return path


def match(face_detect, face_sample):
    requestUrl = 'https://aip.baidubce.com/rest/2.0/face/v3/match'
    token = getToken()
    t = cv2.imread(face_sample, cv2.IMREAD_COLOR)
    path1 = r'D:\temp1.jpg'
    path2 = r'D:\temp2.jpg'
    cv2.imwrite(path1, face_detect)
    cv2.imwrite(path2, t)
    # 二进制方式打开图片文件
    f1 = open(path1, 'rb')
    img1 = b64encode(f1.read())  # base64编码
    f2 = open(path2, 'rb')
    img2 = b64encode(f2.read())  # base64编码
    params = [
        {
            "image": str(img1, 'utf-8'),  # b --> str
            "image_type": "BASE64",
            "face_type": "LIVE",
        },
        {
            "image": str(img2, 'utf-8'),
            "image_type": "BASE64",
            "face_type": "LIVE",
        }
    ]
    # print(params, type(params))
    requestUrl = requestUrl + "?access_token=" + token
    headers = {'content-type': 'application/json'}
    response = requests.post(requestUrl, json=params, headers=headers)
    content = response.json()
    sleep(0.3)
    print(content)
    score = content['result']['score']
    return score


def match_process(face_sample, faces_det):
    for i in range(len(faces_det)):
        highest_score = 0
        for j in range(len(face_sample)):
            score = int(match(faces_det[i].img, face_sample[j]))
            if (score >= 75 and score > highest_score):
                highest_score = score
                faces_det[i].match = 1
                faces_det[i].name = os.path.split(face_sample[j])[1].split('.')[0]
        print("one people over")
    return faces_det


def face_rigester(face_path):
    request_url = "https://aip.baidubce.com/rest/2.0/face/v3/faceset/user/add"
    faces_sample = search(face_path, 'jpg')
    for imgpath in faces_sample:
        f = open(imgpath, 'rb')
        temp = f.read()
        image = b64encode(temp)
        params = {
            "image": image,
            "image_type": "BASE64",
            "group_id": "0",
            "user_id": os.path.split(imgpath)[1].split('.')[0],
            "user_info": os.path.split(imgpath)[1].split('.')[0],
        }
        access_token = getToken()
        request_url = request_url + "?access_token=" + access_token
        headers = {'content-type': 'application/json'}
        response = requests.post(request_url, data=params, headers=headers)
        content = response.json()
        print(content)


def face_detect(face_path, imgpath):
    face_sample = search(face_path, 'jpg')
    images = search(imgpath, 'jpg')
    for image in images:
        faces_det = getData(image)
        faces_det = match_process(face_sample, faces_det)
        draw(image, faces_det)

    # faces_det = getData(images[2])
    # faces_det = match_process(face_sample, faces_det)
    # draw(images[2],faces_det)


def detect(face_path, imgpath):
    face_rigester(face_path)
    images = search(imgpath, 'jpg')
    for image in images:
        faces_det = getData(image)
        draw(image, faces_det)
    face_detect_fast(imgpath)
    delete_userlist()


def face_detect_fast(imgpath):
    images = search(imgpath, 'jpg')
    for image in images:
        request_url = "https://aip.baidubce.com/rest/2.0/face/v3/multi-search"
        f = open(image, 'rb')
        temp = f.read()
        img = b64encode(temp)
        params = {"image": img,
                  "image_type": "BASE64",
                  "group_id_list": "0",
                  "max_face_num": 5,
                  "match_threshold": 75
                  }
        access_token = getToken()
        request_url = request_url + "?access_token=" + access_token
        headers = {'content-type': 'application/json'}
        response = requests.post(request_url, data=params, headers=headers)
        content = response.json()
        print(content)
        face_num = content['result']['face_num']
        img_d = cv2.imread(os.path.split(image)[0] + os.path.split(image)[1].split('.')[0] + 'detect.jpg',
                           cv2.IMREAD_COLOR)
        for i in range(face_num):
            location = content['result']['face_list'][i]['location']
            leftTopX = int(location['left'])
            leftTopY = int(location['top'])
            if (content['result']['face_list'][i]['user_list'] != []):
                name = content['result']['face_list'][i]['user_list'][0]['user_id']
                cv2.putText(img_d, name, (leftTopX, leftTopY - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
            cv2.imwrite(os.path.split(image)[0] + os.path.split(image)[1].split('.')[0] + 'detect.jpg', img_d)


def delete_userlist():
    request_url = "https://aip.baidubce.com/rest/2.0/face/v3/faceset/group/delete"
    params = {"group_id": "0"}
    access_token = getToken()
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/json'}
    requests.post(request_url, data=params, headers=headers)

# face_detect(r'D:\university\robocupface\faces\\',r'D:\university\robocupface\\')

# face_rigester(r'D:\university\robocupface\faces\\')

# face_detect_fast(r'D:\university\robocupface\\')
# detect(r'D:\university\robocupface\faces\\',r'D:\university\robocupface\\')
