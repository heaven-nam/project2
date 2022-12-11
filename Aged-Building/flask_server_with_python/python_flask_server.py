from flask import Flask, jsonify, request
# from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] =


# '@' is decorator
@app.route('/', methods=['GET'])  # '/' (사용자가 어떤 pass를 작성하지 않고 접속한다면? -> index()함수가 응대 합니다.)
def get_articles():
    return jsonify({"Hello":"World!"})

@app.route('/image', method=['POST'])
def get_image():
    # 1. POST로 전달된 json에서 문자열 형태의 이미지 데이터 추출

    # 2. 문자열 형태의 이미지 데이터를 이미지 데이터로 변환

    # 3. Object Detection 수행

    # 4. 결과값을 return (json 형태로)
    pass

if __name__ == "__main__":
    app.run(port=5000, debug=True)
# '5000' is the number of port
# 'debug=True' automatically applies when the code changes.
