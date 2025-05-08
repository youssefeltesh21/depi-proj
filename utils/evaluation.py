import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold

# Import your custom modules - keep these as they are in your code
from recommender.collaborative_filtering import SVDF  # or wherever the class lives
from utils.data_loader import load_cleaned_ratings


def evaluate_SVDF_RMSE(train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       n_factors=100,
                       lr=0.02,
                       reg=0.02,
                       n_epochs=70,
                       random_state=42):
    """
    Fit on ``train_df`` and return RMSE, MSE, MAE on ``test_df``.
    """
    model = SVDF(n_factors=n_factors,
                 lr=lr,
                 reg=reg,
                 n_epochs=n_epochs,
                 random_state=random_state)
    model.fit(train_df)

    y_true = test_df["Book-Rating"].to_numpy()
    y_pred = [model.predict(u, i) for u, i in zip(test_df["User-ID"], test_df["ISBN"])]

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mse, mae


def evaluate_SVDF_precision_at_k(train_df: pd.DataFrame,
                                 test_df: pd.DataFrame,
                                 k: int = 10,
                                 relevance_threshold: int = 7,
                                 n_factors=100,
                                 lr=0.02,
                                 reg=0.02,
                                 n_epochs=70,
                                 random_state=42):
    """
    Evaluates Precision@K for the SVDF model.
    Precision@K = (number of relevant items in top-K recommendations) / K
    """
    model = SVDF(n_factors=n_factors,
                 lr=lr,
                 reg=reg,
                 n_epochs=n_epochs,
                 random_state=random_state)
    model.fit(train_df)

    test_users = test_df["User-ID"].unique()
    precision_scores = []

    for user in test_users:

        user_test_isbn = set(test_df[(test_df["User-ID"] == user) &
                                     (test_df["Book-Rating"] > relevance_threshold)]["ISBN"])
        if len(user_test_isbn) == 0:
            continue

        recommended_items = set(model.recommend(user, N=k, df=train_df))
        n_relevan_items = len(user_test_isbn.intersection(recommended_items))
        precision = n_relevan_items / k
        precision_scores.append(precision)

    avg_precision_at_k = np.mean(precision_scores)
    return avg_precision_at_k


"""
TESTING evaluate_precision_at_k
"""
ratings = load_cleaned_ratings()

train_df, test_df = train_test_split(
ratings,
test_size=0.2,
random_state=42,
stratify=ratings["User-ID"]
)

precision_at_10 = evaluate_SVDF_precision_at_k(train_df, test_df, k=10)
print(f"Precision@10: {precision_at_10:.4f}")

rmse, mse, mae = evaluate_SVDF_RMSE(train_df, test_df)
print(f"{rmse:.1f} | MSE: {mse:.1f} | MAE: {mae:.1f}")