import numpy as np
import pandas as pd
from sqlalchemy.testing.suite.test_reflection import users
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from utils.data_loader import *



ratings_df = load_cleaned_ratings()
books_df = load_cleaned_books()
users_df = load_cleaned_users()
user_item_csr = load_user_item_matrix()
user_id_map,isbn_map = load_mappers()

NUM_USERS = users_df.shape[0]
NUM_BOOKS = books_df.shape[1]
N_LATENT_FACTORS = 100

user_id_map_inv = {i:k for k,i in user_id_map.items()}
isbn_map_inv = {i:k for k,i in isbn_map.items()}


u,s,v = svds(user_item_csr, k=N_LATENT_FACTORS)
s = np.diag(s)

user_latent_features = u @ s
item_latent_features = v.T

