from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

from datetime import datetime, timedelta
import pandas as pd
import math
import re
from collections import Counter

import markdown
import json
import requests
import app_user_keyword.views as userkeyword_views

url = "http://163.18.22.32:11435/api/generate"
# 設置遠程 Ollama 模型的基礎 URL
REMOTE_OLLAMA_URL = "http://163.18.22.32:11435"

model_name = "gemma3:4b"  # 默認模型名稱
# model_name = "qwen2.5:7b"  # 默認模型名稱
#model_name = "deepseek-r1:14b"  # 默認模型名稱
# 列出所有可用的模型
print(f"正在連接 {REMOTE_OLLAMA_URL} 檢查可用模型...")
response = requests.get(f"{REMOTE_OLLAMA_URL}/api/tags")
models = response.json()        
print("\n可用的模型:")
available_models = [model['name'] for model in models['models']]
for model in available_models:
    print(f"- {model}")
# 檢查指定的模型是否可用
if model_name in available_models:
    print(f"\n✅ 模型 '{model_name}' 已可用")

def load_df_data():
    # import and use df from app_user_keyword 
    global df # global variable
    df = userkeyword_views.df

load_df_data()


# For the key association analysis
def home(request):
    return render(request, 'app_user_keyword_summarize/home.html')

# df_query should be global
@csrf_exempt
def api_get_userkey_summarize(request):

    userkey = request.POST.get('userkey')
    cate = request.POST['cate'] # This is an alternative way to get POST data.
    cond = request.POST.get('cond')
    weeks = int(request.POST.get('weeks'))
    key = userkey.split()

    global  df_query # global variable It's not necessary.

    df_query = filter_dataFrame_fullText(key, cond, cate,weeks)
    print(key)
    print(len(df_query))

    if len(df_query) == 0:
        return {'error': 'No results found for the given keywords.'}

    if len(df_query) != 0:  # df_query is not empty
        newslinks = get_title_link_topk(df_query, k=15)
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
def filter_dataFrame_fullText(user_keywords, cond, cate, weeks):

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
    if (cate == "全部"):
        condition = period_condition  # "全部"類別不必過濾新聞種類
    else:
        # category新聞類別條件
        condition = period_condition & (df.category == cate)

    # (3) proceed filtering: news category
    # and or 條件
    if (cond == 'and'):
        # query keywords condition使用者輸入關鍵字條件and
        condition = condition & df.content.apply(lambda text: all(
            (qk in text) for qk in user_keywords))  # 寫法:all()
    elif (cond == 'or'):
        # query keywords condition使用者輸入關鍵字條件
        condition = condition & df.content.apply(lambda text: any(
            (qk in text) for qk in user_keywords))  # 寫法:any()
    # condiction is a list of True or False boolean value
    df_query = df[condition]

    return df_query

# get titles and links from k pieces of news 
def get_title_link_topk(df_query, k=25):
    items = []
    for i in range( len(df_query[0:k]) ): # show only 10 news
        category = df_query.iloc[i]['category']
        title = df_query.iloc[i]['title']
        link = df_query.iloc[i]['link']
        photo_link = df_query.iloc[i]['photo_link']
        # if photo_link value is NaN, replace it with empty string 
        if pd.isna(photo_link):
            photo_link=''
        
        item_info = {
            'category': category, 
            'title': title, 
            'link': link, 
            'photo_link': photo_link
        }

        items.append(item_info)
    return items 

def get_title_content(df_query, k=25):
    datas = []
    for i in range( len(df_query[0:k]) ): # show only 10 news
        title = df_query.iloc[i]['title']
        content = df_query.iloc[i]['content']
 
        data_info = { 
            'title': title, 
            'content': content, 
        }

        datas.append(data_info)
    return datas

@csrf_exempt
def api_get_userkey_report(request):
    mode = request.POST['mode']
    result = api_get_userkey_summarize(request)

    print("模式：", mode)

    if isinstance(result, dict) and 'error' in result:
        return JsonResponse(result)

    # 獲取新聞資料（預設最多25則）
    news_items = get_title_content(df_query, k=25)
    print(len(news_items))

    all_reports = []  # 儲存多段回應的容器

    # ===== 摘要模式：一次處理所有新聞 =====
    if mode == '摘要':
        news_text = ""
        for idx, item in enumerate(news_items):
            news_text += f"### 新聞 {idx+1}\n**標題：** {item['title']}\n**內容：** {item['content']}\n\n"

        prompt = f'''
請幫我總結以下新聞內容，生成一篇專業的報導摘要（約500字），請使用繁體中文，並用 Markdown 語法排版：

{news_text}
        '''

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(url, json=payload, timeout=100)
            result = response.json()
            final_report = result['response']
        except:
            print("Error:", response.status_code, response.text)
            return JsonResponse({'error': 'Failed to generate summary. Please try again later.'})

        return JsonResponse({'report': final_report})

    # ===== 分段模式：處理關鍵問答 / 分析型報導 =====
    batch_size = 2
    for i in range(0, len(news_items), batch_size):  # 從 index 0 開始
        batch = news_items[i:i + batch_size]

        summary_blocks = ""
        news_text = ""
        for idx, item in enumerate(batch):
            news_index = i + idx + 1  # 正確新聞編號
            title = item['title']
            content = item['content']

            summary_blocks += f'''## 📰 新聞 {news_index}：{title}
    ### 重點摘要
    一段新聞摘要

    '''
            news_text += f"### 新聞 {news_index}\n**標題：** {title}\n**內容：** {content}\n\n"

        if mode == '關鍵問答':
            prompt = f'''

    不要生成新聞內容只要下面文字有要求的，且問題1和問題2的格式要相同
    以下是幾則新聞的重點摘要，請你根據這些資訊生成兩個關鍵問題及其對應答案，格式請用 Markdown，像這樣：

    ---

    ## 📰 新聞 1：新聞標題

    ### 重點摘要  
    一段新聞摘要

    ### 🔍 問題與回答

    **問題 1** 問題文字  
     
    **回答：** 回答文字

    **問題 2**  問題文字  
    
    **回答：** 回答文字

    ---

    {summary_blocks}
    {news_text}
        '''


        elif mode == '分析型報導':
            prompt = f'''
    請幫我深入分析以下新聞內容，並針對每則新聞分別撰寫以下內容，字數約 250 字。請使用繁體中文，並用 Markdown 排版，格式如下：

    ---

    ## 📰 新聞 X：新聞標題

    ### 核心議題
    （簡要說明新聞的核心議題）

    ### 利害關係人
    - **利害關係人 A**：說明
    - **利害關係人 B**：說明
    （視需要增減）

    ### 可能的發展趨勢
    - 說明可能的後續發展 1
    - 說明可能的後續發展 2

    ### 合理推論
    - 綜合推論與潛在影響

    ---

    以下是新聞內容：

    {news_text}
        '''


        else:
            return JsonResponse({'error': '不支援的模式'})

        # 發送 API 請求
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(url, json=payload, timeout=100)
            result = response.json()
            all_reports.append(result['response'])
        except:
            print("Error:", response.status_code, response.text)
            return JsonResponse({'error': 'Failed to generate report in segment.'})

    final_report = '\n\n'.join(all_reports)
    return JsonResponse({'report': final_report})



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

    
print("app_user_keyword_summarize was loaded!")
