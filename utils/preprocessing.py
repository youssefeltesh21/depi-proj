import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


def clean_ratings(ratings_df):
    """Clean ratings dataframe by removing zeros and nulls"""
    ratings_df["Book-Rating"] = ratings_df["Book-Rating"].replace(0, np.nan).astype(float)
    return ratings_df.dropna(subset=["Book-Rating"])


def clean_books(books_df):
    """Clean books dataframe by removing nulls and duplicates"""
    books_df.dropna(inplace=True)
    books_df.drop_duplicates(subset="Book-Title", keep="first", inplace=True)
    return books_df


def filter_valid_ratings(ratings_df, books_df, users_df):
    """Filter ratings to only include valid books and users"""
    ratings_df = ratings_df[ratings_df["ISBN"].isin(books_df["ISBN"])]
    ratings_df = ratings_df[ratings_df["User-ID"].isin(users_df["User-ID"])]
    return ratings_df


def align_user_book_datasets(ratings_df, books_df, users_df):
    """Filter books and users to only those present in ratings"""
    books_df = books_df[books_df["ISBN"].isin(ratings_df["ISBN"])]
    users_df = users_df[users_df["User-ID"].isin(ratings_df["User-ID"])]
    return books_df, users_df

def apply_k_core_filtering(ratings_df, min_ratings_user=5, min_ratings_book=7):
    """Iteratively filter sparse users and items using k-core method"""
    while True:
        book_counts = ratings_df["ISBN"].value_counts()
        user_counts = ratings_df["User-ID"].value_counts()

        active_books = book_counts[book_counts > min_ratings_book].index
        active_users = user_counts[user_counts > min_ratings_user].index

        filtered_ratings = ratings_df[
            ratings_df["User-ID"].isin(active_users) &
            ratings_df["ISBN"].isin(active_books)
            ]

        if filtered_ratings.shape[0] == ratings_df.shape[0]:
            break
        ratings_df = filtered_ratings

    return ratings_df


def build_user_item_matrix(ratings_df):
    """Create sparse user-item matrix in CSR format"""
    num_users = len(ratings_df["User-ID"].unique())
    num_books = len(ratings_df["ISBN"].unique())

    user_id_map = dict(zip(np.sort(ratings_df["User-ID"].unique()), range(num_users)))
    isbn_map = dict(zip(np.sort(ratings_df["ISBN"].unique()), range(num_books)))

    user_indices = [user_id_map[uid] for uid in ratings_df["User-ID"]]
    book_indices = [isbn_map[isbn] for isbn in ratings_df["ISBN"]]

    return csr_matrix(
        (ratings_df["Book-Rating"].values, (user_indices, book_indices)),
        shape=(num_users, num_books)
    ), user_id_map, isbn_map
