#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 09:57:55 2019

@author: dariush
"""

# imports
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json
import csv
import re
from time import sleep
from random import randint
  
# Importing socket library 
import socket 
import socks
from urllib import request

##############################

# code block is for changing ip address while scraping SKIP if not needed
# for setup instructions on ubuntu see https://www.linuxuprising.com/2018/10/how-to-install-and-use-tor-as-proxy-in.html
# note can set to obtain new ip at intervals - default interval 10mins

# check the external ip address
r = request.urlopen('http://icanhazip.com')
print(r.read())

#use pysocks to access via tor proxy
socks.set_default_proxy(socks.SOCKS5, "localhost", 9050)
socket.socket = socks.socksocket

# check the external ip again
r = request.urlopen('http://icanhazip.com')
print(r.read())


##############################

# get full text of 1 article - as a test can SKIP
# note article text still has unwanted characters - they can be cleaned after download with last code block

# target URL from the dataset to scrape as a test
url = "https://www.huffpost.com/entry/meryl-streep-harvey-weinstein-me-too_n_5a4ceda3e4b06d1621bc56cc"

# headers
headers = {
    'User-Agent': "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36"
    }

# send request to download the data
page = requests.request("GET", url, headers=headers)

# parse the downloaded data
data = BeautifulSoup(page.text, 'html.parser')

# extract full text of the article
article = data.find_all("div", {"class": "content-list-component yr-content-list-text text"})

print(article)

################################

# get full text of many articles getting urls from dataframe

dataset = pd.read_json('News_Category_Dataset_v2.json', lines=True)

urlList = dataset['link'].tolist()

# headers
headers = {
    'User-Agent': "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36"
    }

#create file to write to
f = csv.writer(open('articlefulltext.csv','w'))
f.writerow(['link','article_text'])

#break up urlList into chuncks - creates chunks of 1,843 urls
urlchunk = np.array_split(np.array(urlList),109)

#loop through url list
for url in urlchunk[1]:
    page = requests.request("GET", url, headers=headers)
    data = BeautifulSoup(page.text, 'html.parser')
    article = data.find_all("div", {"class": "content-list-component yr-content-list-text text"})
    f.writerow([url, article])
    #pause the loop
    #sleep(randint(2,4)) #doesn't help

####################################

# read back the csv to remove unwanted html tags
docs = pd.read_csv('articlefulltext.csv')

docs['article_text'] = docs['article_text'].str.replace(r'<.*?>','')
docs['article_text'] = docs['article_text'].str.replace('[','')
docs['article_text'] = docs['article_text'].str.replace("]","")


#export back to csv
docs.to_csv("articles.csv")