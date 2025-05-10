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


if __name__ == "__main__":
    add_new_user("Abdallah","ab1234")
    add_rating(278844,'059035342X',8)
    add_rating(278844,'0451523415',9)
    add_rating(278844, '0804106304', 8)
    add_rating(278844, '0060928336', 9)

    model = FunkSVD()
    model.fit()

    print(model.recommend(278844))