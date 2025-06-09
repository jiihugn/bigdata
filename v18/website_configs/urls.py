from django.contrib import admin
from django.urls import path
from django.urls import include

urlpatterns = [
    # admin
    path('admin/', admin.site.urls),
    # top keywords
    path('topword/', include('app_top_keyword.urls')),
    # top persons
    path('topperson/', include('app_top_person.urls')),
    # user keyword analysis
    path('userkeyword/', include('app_user_keyword.urls')),
    # fans
    path('fans/', include('app_fans.urls')),
    # v5 full text search and associated keyword display
    path('userkeyword_assoc/', include('app_user_keyword_association.urls')),
    # media top keywords
    path('mediatopkeyword/', include('app_media_top_keyword.urls')),
    # sports
    path('sports/', include('app_sports.urls')),
    # sentiment
    path('sentiment/', include('app_user_keyword_sentiment.urls')),
    # pk
    path('', include('app_pk.urls')),
    # full text search and associated keyword display using db
    path('userkeyword_db/', include('app_user_keyword_db.urls')),
    # full text search and associated keyword display using db
    path('topperson_db/', include('app_top_person_db.urls')),
     # user keyword sentiment 
    path('userkeyword_report/', include('app_user_keyword_llm_report.urls')),
    # sentiment analysis, news classification and qwen2.5-0.5b
    path('llm/', include('app_llm_classifier.urls')),
    # Calling ollama usage
    path('ollama/', include('app_ollama_usage.urls', namespace='app_ollama_usage')),
    # Sentiment classification with bert
    path('sentiment/', include('app_sentiment_bert.urls')),
    # News classification
    path('news_cls/', include('app_news_classification_bert.urls')),
    path('llm_intro/', include('app_llm_introduction.urls')),
    # user keyword summarize
    path('userkeyword_summarize/', include('app_user_keyword_summarize.urls', namespace='app_user_keyword_summarize')),
    path('intro/', include('app_intro.urls')),
]
