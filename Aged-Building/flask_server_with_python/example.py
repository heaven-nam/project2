from flask import Flask, request, jsonify
import json
import base64
import os
from PIL import Image
from io import BytesIO
import numpy as np
# import wp_utils    [파일 이름] make image feature json, image rotate
# import detect_text [파일 이름] find and crop text box
# import text_recog  [파일 이름] text-recognition from croped images
# import shape_claaisfication [파일이름]

app = Flask(__name__)

# [알약] @app.rout('/users')
# def users():
# # users 데이터를 json 형식으로 반환한다.
# return {"members": [{"id":1, "name": "yerin"}, {"id" : 2, "name" : "dalkong" }]}

app.config['JSON_AS_ASCII'] = False

@app.route("/", methods=['POST'])
def get_json():
    # 1. Json으로 데이터를 받는다.
    params = request.get_json()
    pill_imag = decode_image(params)

    # 2. 이미지가 저장 되었음을 확인 한다.
    if (pill_imag is not None):
        crop_files = detct_text.

        # 3. 알약의 특징 정보를 json 파일로 저장 한다.
        return "Hello, world"

@app.errorhandler(500)
def error_500():
    return jsonify({"is_success": False, "message": "Server not work"}, 500)

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"is_success":False, "message": "page not found"}, 404)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")