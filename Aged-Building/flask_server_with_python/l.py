from flask import Flask

# import random # python built-in module

app = Flask(__name__)

# 보통 DB에 저장 된다.
topics = [
    # <li>
    {'id': 1, 'title': 'html', 'body': 'html is...'},
    # <li>
    {'id': 2, 'title': 'css', 'body': 'cssl is...'},
    # <li>
    {'id': 3, 'title': 'javascript', 'body': 'javascript is...'}
]


def template(contents, content):
    return f'''<!doctype html>
       <html>
           <body>
               <h1><a href="/">WEB</a></h1>
               <ol>
                   {contents}
               </ol>
               {content}
           </body>
       </html>
       '''


def get_contents():
    liTags = ''
    for topic in topics:
        liTags = liTags + f'<li><a href="/read/{topic["id"]}/">{topic["title"]}</a></li>'
    return liTags


# '@' is decorator
@app.route('/')  # '/' (사용자가 어떤 pass를 작성하지 않고 접속한다면? -> index()함수가 응대 합니다.)
def index():
    return template(get_contents(), '<h2>Welcome</h2>Hello, WEB')


@app.route('/create/')
def create():
    return 'Create'


@app.route('/read/<int:id>/') # <parameter>: 무조건 문자열로 만들어 주기 때문에, int를 넣어 '자동 변환'기능을 넣어 줍니다.
def read(id): # (parameter)
    title = ''
    body = ''
    for topic in topics:
        if id == topic['id']:
            title = topic['title']
            body = topic['body']
            break
    return template(get_contents(), f'<h2>{title}</h2>{body}')


app.run(port=5003, debug=True)
# '5000' is the number of port
# 'debug=True' automatically applies when the code changes.
