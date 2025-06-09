from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

from datetime import datetime, timedelta
import pandas as pd
import math
import re
from collections import Counter

import app_user_keyword.views as userkeyword_views
def load_df_data():
    # import and use df from app_user_keyword 
    global df # global variable
    df = userkeyword_views.df

load_df_data()


# For the key association analysis
def home(request):
    return render(request, 'app_sports/home.html')

# df_query should be global
@csrf_exempt
def api_sports(request):

    sport = request.POST['sport'] # This is an alternative way to get POST data.
    weeks = int(request.POST.get('weeks'))

    #global  df_query # global variable It's not necessary.

    df_query = filter_dataFrame_fullText(sport,weeks)
    print(sport)
    print(len(df_query))

    if len(df_query) != 0:  # df_query is not empty
        newslinks = get_title_link_topk(sport,df_query, k=15)
        related_words, clouddata = get_related_word_clouddata(df_query)

    else:
        newslinks = []
        related_words = []
        clouddata = []

    response = {
        'newslinks': newslinks,
        'related_words': related_words,
        'clouddata':clouddata,
        'num_articles': len(df_query),
    }
    return JsonResponse(response)


# Searching keywords from "content" column
# Here this function uses df.content column, while filter_dataFrame() uses df.tokens_v2
def filter_dataFrame_fullText(sport, weeks):

    # end date: the date of the latest record of news
    end_date = df.date.max()

    # start date
    start_date = (datetime.strptime(end_date, '%Y-%m-%d').date() -
                  timedelta(weeks=weeks)).strftime('%Y-%m-%d')

    # (1) proceed filtering: a duration of a period of time
    # 期間條件
    period_condition = (df.date >= start_date) & (df.date <= end_date)

    # (2) proceed filtering: news category
    # 新聞類別條件
    if (sport == "棒球"):
        sport = ['棒球','中職', 'MLB','大聯盟','職棒','日職']
    elif (sport == "籃球"):
        sport = ['籃球', 'NBA']
    elif (sport == "足球"):
        sport = ['足球', '英超', '西甲', '德甲', '意甲', '法甲']
    elif (sport == "羽球"):
        sport = ['羽球']

    if (sport == "全部"):
        condition = period_condition & (df.category == '運動')   # "全部"類別不必過濾新聞種類
    else:
        # category新聞類別條件
        condition = period_condition & (df.category == '運動') & df.content.apply(lambda text: any(
            (qk in text) for qk in sport)) 

    # condiction is a list of True or False boolean value
    df_query = df[condition]

    return df_query


# get titles and links from k pieces of news 
def get_title_link_topk(sport,df_query, k=25):
    items = []
    for i in range( len(df_query[0:k]) ): # show only 10 news
        sports = sport
        title = df_query.iloc[i]['title']
        link = df_query.iloc[i]['link']
        photo_link = df_query.iloc[i]['photo_link']
        # if photo_link value is NaN, replace it with empty string 
        if pd.isna(photo_link):
            photo_link=''
        
        item_info = {
            'sports': sports, 
            'title': title, 
            'link': link, 
            'photo_link': photo_link
        }

        items.append(item_info)
    return items 

# Get related keywords by counting the top keywords of each news.
# Notice:  do not name function as  "get_related_keys",
# because this name is used in Django
def get_related_word_clouddata(df_query):

    # wf_pairs = get_related_words(df_query)
    # prepare wf pairs 
    counter=Counter()
    for idx in range(len(df_query)):
        pair_dict = dict(eval(df_query.iloc[idx].top_key_freq))
        counter += Counter(pair_dict)
    wf_pairs = counter.most_common(20) #return list format

    # cloud chart data
    # the minimum and maximum frequency of top words
    min_ = wf_pairs[-1][1]  # the last line is smaller
    max_ = wf_pairs[0][1]
    # text size based on the value of word frequency for drawing cloud chart
    textSizeMin = 20
    textSizeMax = 120
    if (min_ != max_):
        max_min_range = max_ - min_

    else:
        max_min_range = len(wf_pairs) # 關鍵詞的數量: 20個
        min_ = min_ - 1 # every size is 1 / len(wf_pairs)
    
    # word cloud chart data using proportional scaling
    # 排除分母為0的情況
    clouddata = [{'text':w, 'size':int(textSizeMin + (f - min_)/max_min_range * (textSizeMax-textSizeMin))} for w, f in wf_pairs]


    return   wf_pairs, clouddata 


print("app_sports was loaded!")
