from django.http import JsonResponse
from django.shortcuts import render
import pandas as pd

def load_data_fans():
    # Read data from csv file
    df_data = pd.read_csv('app_fans/dataset/something_data.csv',sep=',')
    global response
    response = dict(list(df_data.values))
    del df_data

# load data
load_data_fans()

#print(response)

def home(request):
    return render(request,'app_fans/home.html', response)

print('app_fans was loaded!')
