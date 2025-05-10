import csv
import numpy as np
import pandas as pd

from recommender.collaborative_filtering import FunkSVD
from utils.data_loader import *
from utils.preprocessing import build_mappers

users_df = load_cleaned_users()
ratings_df = load_cleaned_ratings()


def add_new_user(user_name,password,
                 user_id = users_df.iloc[-1,0] + 1,
                 loc = "Unknown", Age = None,
                 path = r"..\data\processed\users_cleaned.csv",
                 input_df = users_df):

    """
    Adding a new user to user's dataset
    """

    if (user_name in input_df["User-Name"].values) or (user_id in input_df.values):
        print("User already exists")
        return

    if len(password) < 4:
        print("Password must be at least 4 characters")
        return

    new_user_row = [user_id,user_name,loc,Age,password]
    with open(path, "a",newline='') as f:
        writer = csv.writer(f)
        writer.writerow(new_user_row)
        print("User added successfully")

    input_df.iloc[-1] = new_user_row
    build_mappers(load_cleaned_ratings())


def add_rating(user_id : int,
               isbn : str,
               rating : int,
               path = r"..\data\processed\ratings_cleaned.csv",
               input_df = ratings_df):

    """
    Appending a new rating row to ratings_cleaned.csv
    """

    user_isbns_df = ratings_df.groupby("User-ID")["ISBN"].unique()

    if (user_id in user_isbns_df) and (isbn in user_isbns_df.loc[user_id]):
        print("User already rated that book")
        return

    new_rating = [int(user_id),str(isbn),float(rating)]

    with open(path, "a",newline='') as f:
        writer = csv.writer(f)
        writer.writerow(new_rating)
        print("Rating added successfully")

    input_df.iloc[-1] = new_rating
    build_mappers(load_cleaned_ratings())


def expand_model_for_new_users(model, ratings_df, save_path='../models/collaborative_filtering_model.pkl'):
    """Update an existing model to handle new users"""
    new_maps = load_mappers()
    model.user_id_map = new_maps[0]
    model.isbn_map = new_maps[1]
    model.user_id_map_inv = new_maps[2]
    model.isbn_map_inv = new_maps[3]

    # Expand P matrix for new users if needed
    old_user_count = model.P.shape[0]
    new_user_count = len(model.user_id_map)

    if new_user_count > old_user_count:
        # Create P vectors for new users
        new_rows = model.rng.normal(0, 0.1, (new_user_count - old_user_count, model.k))
        model.P = np.vstack([model.P, new_rows])

    return model


def personalize_model_for_user(model, user_id, ratings_df,
                                save_path='../models/collaborative_filtering_model.pkl'):

    """Train factors just for a specific user"""
    user_ratings_df = ratings_df[ratings_df["User-ID"] == user_id]
    user_idx = model.user_id_map[user_id]
    isbn_indices = [model.isbn_map[isbn] for isbn in user_ratings_df["ISBN"].values]
    ratings_values = user_ratings_df["Book-Rating"].values

    for epoch in range(model.n_epochs):
        for isbn, rating in zip(isbn_indices, ratings_values):
            pred = model.P[user_idx] @ model.Q[isbn]
            err = rating - pred

            model.P[user_idx] = model.P[user_idx] + model.lr * (err * model.Q[isbn] - model.reg * model.P[user_idx])

    return model

def save_trained_model(model, path  = '../models/collaborative_filtering_model.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path} successfully")

'''## Usage:
if __name__ == "__main__":
    ratings_df = load_cleaned_ratings()
    users_df = load_cleaned_users()

    add_new_user("Abdallah", "ab1234",input_df = users_df)
    add_rating(278844, '059035342X', 8,input_df= ratings_df)
    add_rating(278844, '0804106304', 8,input_df= ratings_df)
    add_rating(278844, '0743412028', 8,input_df= ratings_df)
    add_rating(278844, '0380002450', 9,input_df= ratings_df)

    model = load_trained_model()
    model = expand_model_for_new_users(model,ratings_df)
    model = personalize_model_for_user(model, 278844, ratings_df)

    print(model.recommend(278844))
    save_trained_model(model)'''