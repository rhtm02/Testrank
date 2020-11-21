from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def matching_calculator(investor_keyword, company, mode = 0):
    # investor_keyword is just one investor & keyword list data
    # company is all company & dataframe data
    # mode 0 is investor -> company 1 is company -> investor
    if(mode == 0):
        company_id_list = company['company_id']
    else:
        company_id_list = company['investor_id']
    company_doc_list = []
    investor_doc_list = [' '.join(investor_keyword)]

    # transform company ketword to string
    for keyword in company['keyword']:
        company_doc_list.append(' '.join(keyword))
    # merge company, investor
    analysis_data = investor_doc_list + company_doc_list
    # tf-idf
    TF_IDF =  TfidfVectorizer()
    feature_vect_simple = TF_IDF.fit_transform(analysis_data)

    cosine_score = cosine_similarity(feature_vect_simple[0] , feature_vect_simple[1:])[0]
    id_score_dictionary = dict(zip(company_id_list ,cosine_score))
    id_score_pair = sorted(id_score_dictionary.items(), key = lambda item: item[1] ,reverse=True)[0]

    return id_score_pair


def matching_func(investor_df, company_df):
    investor_id_list = investor_df['investor_id']
    company_id_list = company_df['company_id']
    investor_screen = 'matching {} investor {} company, score : {}'
    company_screen = 'matching {} company {} investor, score : {}'
    investor_company_matching = []
    company_investor_matching = []
    for i in range(len(investor_df)):
        company, score = matching_calculator(investor_df['keyword'][i], company_df, mode=0)

        if (score == 0):
            investor_company_matching.append([investor_id_list[i], np.nan, np.nan, '2020-11-25 00:00:00'])
            print(investor_screen.format(investor_id_list[i], np.nan, np.nan))
        else:
            investor_company_matching.append([investor_id_list[i], company, int(score * 1000), '2020-11-25 00:00:00'])
            print(investor_screen.format(investor_id_list[i], company, int(score * 1000)))

    for i in range(len(company_df)):
        investor, score = matching_calculator(company_df['keyword'][i], investor_df, mode=1)

        if (score == 0):
            company_investor_matching.append([company_id_list[i], np.nan, np.nan, '2020-11-25 00:00:00'])
            print(company_screen.format(company_id_list[i], np.nan, np.nan))
        else:
            company_investor_matching.append([company_id_list[i], investor, int(score * 1000), '2020-11-25 00:00:00'])
            print(company_screen.format(company_id_list[i], investor, int(score * 1000)))

    investor_company_matching_df = pd.DataFrame(investor_company_matching,
                                                columns=['investor_id', 'company_id', 'score', 'regdate'])
    company_investor_matching_df = pd.DataFrame(company_investor_matching,
                                                columns=['company_id', 'investor_id', 'score', 'regdate'])

    return investor_company_matching_df, company_investor_matching_df