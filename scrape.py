import numpy as np 
from bs4 import BeautifulSoup 
from requests import get

url = 'https://www.imdb.com/chart/top/?ref_=nv_mv_250'
html = get(url)
soup = BeautifulSoup(html.content, features='html.parser')

table = soup.find('tbody')
movies = table('td', class_ = 'titleColumn')
urls = ['http://imdb.com' + i.find('a').get('href') for i in movies]

url = urls[0]
html = get(url)
soup = BeautifulSoup(html.content, features='html.parser')

title = soup.find('div', class_='title_wrapper')
title.find('h1').text.split('\xa0')

summary = soup.find('div', class_='plot_summary')
summary.find('div', class_='summary_text').text

[i.text for i in soup('div', class_='credit_summary_item')]

# Can we predict how controversial a movie is?
# What determines a male or female centric movie?

