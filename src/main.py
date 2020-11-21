import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import pandas as pd
import re
import os

from matching import *
from textrank import TextRank

print('###############SETTING DIRETORY STRUCTURE###############')
path = "../result"
if not os.path.isdir(path):
    os.mkdir(path)
print('###############COMPLETE###############')

print('###############LOAD DATA###############')
stopwords_csv = pd.read_csv('../data/stopwords.csv')
stopwords_csv = stopwords_csv[stopwords_csv.columns[1:]]
stopwords = stopwords_csv.values.tolist()
stopwords = list(np.asarray(stopwords).reshape(1,-1)[0])
investor_raw = pd.read_csv('../data/investor_list_1차.csv',encoding='cp949')
company_raw = pd.read_csv('../data/company_list_1차.csv',encoding='cp949')
invest_data_raw = investor_raw[['investor_id','investor_category']]
company_data_raw = company_raw[['company_id','company_info','company_category','company_product']]
invest_data_raw = invest_data_raw.fillna(0)
company_data_raw = company_data_raw.fillna(0)

print('###############EXTRACT COMPANY FEATURES###############')
#keyword_format = '{} id, keyword : {}'
df_data = []
for i in range(len(company_data_raw)):
    company_id = company_data_raw['company_id'][i]
    info = company_data_raw['company_info'][i]
    category = company_data_raw['company_category'][i]
    product = company_data_raw['company_product'][i]
    if(info == 0):
        info = ""
    if(category == 0):
        category = ""
    if(product == 0):
        product = ""
    data = info + category + product
    characters = '[-①②③,.#()_총/?\n를을의및에은는로가:;]'
    url = re.sub(characters,'',data)
    try:
        textrank = TextRank(url,stopwords)
        keyword = []
        for j in textrank.ketword():
            keyword.append(j.lower())
        #print(i,keyword_format.format(company_id, keyword[:10]))
        df_data.append([company_id, keyword[:10],'2020-11-23 00:00:00'])
    except:
        keyword = []
        for j in url.split(' '):
            if(j == ''):
                continue
            keyword.append(j.lower())
        #print(i,keyword_format.format(company_id,keyword[:10]))
        df_data.append([company_id, keyword[:10],'2020-11-23 00:00:00'])
print('###############MAKE COMPANY KEYWORD FILE&SAVE###############')
company_df = pd.DataFrame(df_data,columns=['company_id','keyword','regdate'])
company_df.to_csv('../result/company_anal_dic.csv',encoding='cp949')
print('###############COMPANY KEYWORD FILE SAVE TO ../result/company_anal_dic.csv###############')
print('###############EXTRACT INVESTOR FEATURES###############')
df_data = []
for i in range(len(invest_data_raw)):
    investor_id = invest_data_raw['investor_id'][i]
    data = invest_data_raw['investor_category'][i]
    #if nan
    if(data == 0):
        data = ""
    characters = '[-①②③,.#()_총/?\n를을의및에은는로123가ㅇ:;]'
    url = re.sub(characters,'',data)
    try:
        textrank = TextRank(url,stopwords)
        keyword = []
        for j in textrank.ketword():
            keyword.append(j.lower())
        #print(i,keyword_format.format(investor_id, keyword[:10]))
        df_data.append([investor_id, keyword[:10],'2020-11-23 00:00:00'])
    except:
        keyword = []
        for j in url.split(' '):
            if(j == ''):
                continue
            keyword.append(j.lower())
        #print(i,keyword_format.format(investor_id,keyword[:10]))
        df_data.append([investor_id, keyword[:10],'2020-11-23 00:00:00'])

print('###############MAKE INVESTOR KETWORD FILE&SAVE###############')
investor_df = pd.DataFrame(df_data,columns=['investor_id','keyword','regdate'])
investor_df.to_csv('../result/investor_anal_dic.csv',encoding='cp949')
print('###############INVESTOR KEYWORD FILE SAVE TO ../result/investor_anal_dic.csv###############')
print('###############MATCHING###############')
investor,company = matching_func(investor_df,company_df)

print('###############SAVE FILE###############')
investor.to_csv('../result/anal_result2.csv',encoding='cp949')
print('###############INVESTOR MATCHING FILE SAVE TO ../result/anal_result2.csv###############')
company.to_csv('../result/anal_result1.csv',encoding='cp949')
print('###############COMPANY MATCHING FILE SAVE TO ../result/anal_result1.csv###############')







