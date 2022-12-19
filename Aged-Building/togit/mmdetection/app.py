from flask import Flask, jsonify, request, abort, render_template
import cv2
import json
import base64
import io                    
import numpy as np
from PIL import Image
from predictionapp import detc

# import model_for_flask


app = Flask(__name__)

@app.route('/', methods=['GET'])  
def get_articles():
    return 'Hello, We Are B-Antiaging!'

@app.route('/image/', methods=["POST"])
def load_image():
    params = json.loads(request.get_data(), encoding='utf-8')
    print(params.keys())
    print('post coming in')
    im_b64 = request.json['image']
    im_b64 = im_b64.encode('utf-8')
    
    img = post_image(im_b64)

    config = './swinretina_all.py'
    chkpt = 'epoch_24.pth'

    result = detc(config, img, chkpt)

    return json.dumps(result)


def post_image(im_b64):
    # 1. POST로 전달된 json에서 문자열 형태의 이미지 데이터 추출

    # 2. 문자열 형태의 이미지 데이터를 이미지 데이터로 변환

    # 3. Object Detection 수행

    # 4. 결과값을 return (json 형태로)


        #im_b64 = request.json['image'] #json으로 받음
        img_bytes = base64.b64decode(im_b64) 
        img_to_load  = Image.open(io.BytesIO(img_bytes)) 

        #이미지 형태를 확인하기 위해 넣었습니다
        img_arr = np.asarray(img_to_load)
        print('img shape', img_arr.shape)
        
        result_dict = {'output': 'output_key'}

        return img_arr


# def encode_image(result_img_name):

    
#     files = open(result_img_name, 'rb')
#     im_bytes = files.read()
#     img = base64.b64encode(im_bytes).decode("utf8")
    #headers = {'Content-Type': "application/json", 'charset':'utf-8', 'Accept': 'text/plain'}
    #api = 'http://127.0.0.1:5000/image/'

    #payload = json.dumps({"image": img, "other_key": "value"})
    #response = requests.post(api, data=payload, headers=headers)
    #try:
    #    data = response.json()     
    #    print(data)                
    #except requests.exceptions.RequestException:
    #    print(response.text)

    

#이제 object detection 해야해

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='7777', debug=True)
# '5000' is the number of port
