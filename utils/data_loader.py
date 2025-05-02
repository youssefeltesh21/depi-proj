import pandas as pd
import os

def load_books():
    path = r'..\data\raw\Books.csv'
    books = pd.read_csv(path,sep=';',on_bad_lines='skip',encoding = 'latin-1', low_memory=False)
    return books

def load_users():
    path = r'..\data\raw\Users.csv'
    users = pd.read_csv(path,sep=';',on_bad_lines='skip',encoding='latin-1')
    return users

def load_ratings():
    path = r'..\data\raw\Ratings.csv'
    ratings = pd.read_csv(path,sep=';',on_bad_lines='skip',encoding='latin-1')
    return ratings


