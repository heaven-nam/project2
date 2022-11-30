from requests import get
from bs4 import BeautifulSoup
import re

class Crawler:

    def __init__(self):
        url_ = 'https://link.springer.com/search/page/2?facet-language=%22En%22&facet-discipline=%22Mathematics%22' \
               '&query= '
        query = 'complex+analysis&facet-content-type=%22Article%22'

    def crawler(self):
        headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'}

