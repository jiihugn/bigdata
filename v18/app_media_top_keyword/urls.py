from django.urls import path
from app_media_top_keyword import views

app_name="app_media_top_keyword"

urlpatterns = [

    path('', views.home, name='home'),
    path('api_get_media_top_keyword/', views.api_get_media_top_keyword),

]
