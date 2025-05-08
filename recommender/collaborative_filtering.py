import numpy as np
import pandas as pd
from utils.data_loader import *

ratings = load_cleaned_ratings()

class SVDF:

    def __init__(self, n_factors = 100 , lr = 0.02, reg = 0.02, n_epochs = 70, random_state = 42):
        self.k = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.rng = np.random.default_rng(random_state)


    def fit(self,df = ratings):
        maps = load_mappers()
        self.user_id_map = maps[0]
        self.isbn_map = maps[1]
        self.user_id_map_inv = maps[2]
        self.isbn_map_inv = maps[3]
        NUMBER_OF_USERS = len(self.user_id_map)
        NUMBER_OF_BOOKS = len(self.isbn_map)


        self.P = self.rng.normal(0,0.1,(NUMBER_OF_USERS, self.k))
        self.Q = self.rng.normal(0,0.1,(NUMBER_OF_BOOKS,self.k))

        users = np.asarray([self.user_id_map[u] for u in df["User-ID"]])
        isbns = np.asarray([self.isbn_map[isbn] for isbn in df["ISBN"]])
        ratings = df["Book-Rating"].to_numpy()


        for epoch in range(self.n_epochs):
            idx = self.rng.permutation(len(ratings))
            for u, i, rating in zip(users[idx], isbns[idx], ratings[idx]):

                pred = self.P[u] @ self.Q[i]
                err = rating - pred

                P_old = self.P[u].copy()
                self.P[u] = self.P[u] + self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] = self.Q[i] + self.lr * (err * P_old - self.reg * self.Q[i])

        return self


    def predict(self, user_id, isbn):
        user_indx = self.user_id_map[user_id]
        item_indx = self.isbn_map[isbn]

        pred_rating = self.P[user_indx] @ self.Q[item_indx]

        if pred_rating > 10:
            pred_rating = 10
        elif pred_rating < 0:
            pred_rating = 0

        return float(pred_rating)


    def recommend(self, user_id, N = 10,df = ratings):
        user_indx = self.user_id_map[user_id]
        scores = self.Q @ self.P[user_indx]

        seen_isbns = df[df["User-ID"] == user_id]["ISBN"]
        seen_indices = [self.isbn_map[k] for k in seen_isbns]
        scores[seen_indices] = -np.inf

        top_n_scores_indx = np.argsort(scores)[-N:]
        recommended_isbns = [self.isbn_map_inv[i] for i in top_n_scores_indx]
        return recommended_isbns[::-1]