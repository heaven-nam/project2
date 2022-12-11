from flask import Flask

# import random # python built-in module

app = Flask(__name__)


# '@' is decorator
@app.route('/')  # '/' (사용자가 어떤 pass를 작성하지 않고 접속한다면? -> index()함수가 응대 합니다.)
def index():
    # str(random.random()) # return 은 기본적으로 문자열을 응답 해야 한다.
    return 'hye'


@app.route('/create/')
def create():
    return 'Create'


@app.route('/read/1/')
def read():
    return 'Read 1'


app.run(port=5000, debug=True)
# '5000' is the number of port
# 'debug=True' automatically applies when the code changes.
