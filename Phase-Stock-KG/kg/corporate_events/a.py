from requests_html import AsyncHTMLSession
from requests_html import HTMLSession
from bs4 import BeautifulSoup

headers = {
        'User-Agent' :'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Referer' :'https://seekingalpha.com/symbol/AAPL/news?filter=dividend_news&page=1',
        'sec-ch-ua': '"Not_A Brand";v="99", "Google Chrome";v="109", "Chromium";v="109"'
    }
def func():
    
    session = HTMLSession()
    url = 'https://seekingalpha.com/symbol/AAPL/news?filter=dividend_news&page=1'
    r = session.get(url, headers=headers)
    print(r)
    r.html.render(sleep=10)
    return r

r = func()
soup = BeautifulSoup(r.html.raw_html, 'html.parser')

w = open("b.html", "w")
w.write(soup.prettify())
w.close()

