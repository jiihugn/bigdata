from django.urls import path
from . import views

app_name='app_pk'

urlpatterns = [
    path('', views.home, name='home'),
    path('api_get_pk/', views.api_get_pk),
]
