import requests
import json
import base64
import cv2

img = cv2.imread('S-210822_H_F_3_R_96367347051-1.jpg')
img = base64.b64encode(img).decode('utf8')
data = dict()
data['image'] = '1234'
data_json = json.dumps(data)

# r = requests.post('http://192.168.1.185:7777', data=data_json)
r = requests.get('http://127.0.0.1:7777', params = data_json)
print(r)