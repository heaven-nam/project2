from requests import get
from bs4 import BeautifulSoup
import re

url = 'https://scholar.google.com/scholar'
headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'}

v