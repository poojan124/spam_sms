'''
    prepro.py - python code to clean and transform text data for use in classifier
'''

import numpy as np
import pandas as pd
import re
import bs4 as bs
import warnings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class preprocess(object):

    def __init__(self, file_='NONE', cols=[], stop_flag=True):
        self.file_ = file_
        self.stop_flag = stop_flag
        self.cols = cols
        self.cntr = 0
        if stop_flag:
            self.stop_words = set(stopwords.words('english'))

    def stop_remover(self,s):
        new_s = ''
        words = word_tokenize(s)
        for w in words:
            if w.lower() not in self.stop_words:
                new_s = new_s + ' ' + w.lower()
        return new_s[1:]

    # -*- coding: utf-8 -*-
    def isEnglish(self,s):
        try:
            s.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True

    def collapse_terms(self,s):
        for w in word_tokenize(s):
            if (w.find('http')!=-1):
                s = s.replace(w,"_link_feature")
            if self.isEnglish(w) == False:
                s = s.replace(w,"")
        return s

    def read_and_clean(self):
        '''
            read data into pandas dataframe
            keep only columns that are required
        '''
        file = open('data','r')
        x = []
        y = []
        for line in file:
            if line[0]=='h':
                y.append(0)
                x.append(line[3:].lstrip())
            else:
                y.append(1)
                x.append(line[4:].lstrip())

        self.data = pd.DataFrame({'CONTENT':x , 'CLASS':y})
        print("Data Shape : " ,end=' ')
        print(self.data.shape)

        '''
            parse text using html decoder because some of the text are in html format.
        '''
        warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
        self.data['CONTENT'] = self.data['CONTENT'].apply(lambda x: bs.BeautifulSoup(x,'html.parser').get_text())

        '''
            remove punctuation
            remove stop words if stop_words flag is true
        '''
        self.data['CONTENT'] = self.data['CONTENT'].apply(lambda x : re.sub('[^A-Za-z ]','',x))
        if self.stop_flag:
            self.data['CONTENT'] = self.data['CONTENT'].apply(lambda x : self.stop_remover(x))
        '''
            convert
        '''
        '''
            collapse some of the feature which contains links into one commom term
            also remove non english words.
        '''
        self.data['CONTENT'] = self.data['CONTENT'].apply(lambda x :self. collapse_terms(x))
        #self.data['CLASS'] = self.data['CLASS'].apply(lambda x : 1 if x == 'spam' else 0)
        return self.data

if __name__ == "__main__":
    # Testing area
    obj1 = preprocess(['data'],['CLASS','CONTENT'])
    data = obj1.read_and_clean()
    print(obj1.data.head())
