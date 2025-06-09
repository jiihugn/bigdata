from django.urls import path
from app_user_keyword_summarize import views

app_name="app_user_keyword_summarize"

urlpatterns = [

    path('', views.home, name='home'),
    path('api_get_userkey_summarize/', views.api_get_userkey_summarize),
    path('api_get_userkey_report/', views.api_get_userkey_report, name='api_get_userkey_report'),

]
