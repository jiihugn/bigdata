from django.urls import path
from app_sports import views

app_name="app_sports"

urlpatterns = [

    path('', views.home, name='home'),
    path('api_sports/', views.api_sports),

]
