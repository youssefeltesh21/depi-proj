import pandas as pd
from scipy.sparse import load_npz
import json
import os
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent.resolve())
sys.path.append(parent_dir)

import dill


def load_books(path = 'data/raw/Books.csv'):
    books = pd.read_csv(path,sep=';',on_bad_lines='skip',encoding = 'latin-1', low_memory=False)
    return books

def load_users(path = 'data/raw/Users.csv'):
    users = pd.read_csv(path,sep=';',on_bad_lines='skip',encoding='latin-1')
    return users

def load_ratings(path = 'data/raw/Ratings.csv'):
    ratings = pd.read_csv(path,sep=';',on_bad_lines='skip',encoding='latin-1')
    return ratings

def load_cleaned_books(path = 'data/processed/books_cleaned.csv'):
    return pd.read_csv(path)

def load_cleaned_users(path = 'data/processed/users_cleaned.csv'):
    return pd.read_csv(path)

def load_cleaned_ratings(path = 'data/processed/ratings_cleaned.csv'):
    return pd.read_csv(path)

def load_user_item_matrix(path = 'data/processed/user_item.npz'):
    return load_npz(path)

def load_mappers(path = 'data/processed/mappers.json'):
    with open(path,'r') as f:
        json_mappers = json.load(f)
        raw_user_id_map = json_mappers.get('user_id_map')
        isbn_map = json_mappers.get('isbn_map')
        raw_user_id_map_inv = json_mappers.get('user_id_map_inv')
        isbn_map_inv = json_mappers.get('isbn_map_inv')

        user_id_map = {int(k): int(v) for k, v in raw_user_id_map.items()}
        user_id_map_inv = {int(k): int(v) for k, v in raw_user_id_map_inv.items()}
        isbn_map = {k: int(v) for k, v in isbn_map.items()}
        isbn_map_inv = {int(k): v for k, v in isbn_map_inv.items()}

    return user_id_map, isbn_map, user_id_map_inv, isbn_map_inv

def load_trained_model(path=None):
    from recommender.collaborative_filtering import FunkSVD
    if path is None:
        path = 'models/collaborative_filtering_model.pkl'
    try:
        with open(path, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise