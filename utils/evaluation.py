import numpy as np
import pandas as pd
from executing.executing import TESTING
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold

from recommender.collaborative_filtering import SVDF   # or wherever the class lives
from utils.data_loader import load_cleaned_ratings


def evaluate_SVDF(train_df: pd.DataFrame,
                   test_df: pd.DataFrame,
                   n_factors=30,
                   lr=0.005,
                   reg=0.02,
                   n_epochs=20,
                   random_state=42):
    """
    Fit on ``train_df`` and return RMSE, MSE, MAE on ``test_df``.
    """

    # 1) Train
    model = SVDF(n_factors=n_factors,
                 lr=lr,
                 reg=reg,
                 n_epochs=n_epochs,
                 random_state=random_state).fit(train_df)

    # 2) Predict the held-out ratings
    y_true = test_df["Book-Rating"].to_numpy()
    y_pred = [model.predict(u, i)  # single prediction function you wrote
              for u, i in zip(test_df["User-ID"], test_df["ISBN"])]

    # 3) Metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mse, mae

'''
"""
TESTING evaluate_SVDF
"""
ratings = load_cleaned_ratings()

train_df, test_df = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42,
    stratify=ratings["User-ID"]      # keeps user distribution roughly stable
)

rmse, mse, mae = evaluate_SVDF(train_df, test_df)
print(f"Hold-out 20% – RMSE: {rmse:.4f} | MSE: {mse:.4f} | MAE: {mae:.4f}")
'''