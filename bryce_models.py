import torch

from preprocessing import get_full_timeseries_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import PredictionErrorDisplay
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
import pathlib
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import SimpleExpSmoothing

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

import warnings

ELO_CUTOFF = 2500
LSTM_CACHE_DIRECTORY = pathlib.Path("cache") / f"lstm_models_{ELO_CUTOFF}"
LSTM_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)

PLOT_DIR = pathlib.Path("plots")
LINEAR_MODEL_PLOT_DIR = PLOT_DIR / f"linear_models_{ELO_CUTOFF}"
LINEAR_MODEL_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Data preprocessing
def yearly_first_data_subset(df: pd.DataFrame):
    return df.loc[df.groupby(["fideid", df["time"].dt.year])["time"].idxmin()].sort_values(["fideid", "time"]).reset_index(drop=True)

def get_timeseries_data_table(elo_threshold=ELO_CUTOFF):
    return yearly_first_data_subset(get_full_timeseries_model(elo_threshold)[f"All ever > {elo_threshold} players"])

def split_timeseries_data_table_into_train_and_test(df: pd.DataFrame, test_ratio=0.2, validation_ratio = 0.2):
    fideids = df["fideid"].unique()
    np.random.Generator.shuffle(np.random.default_rng(42), fideids)

    split_index = int(len(fideids) * (1 - test_ratio - validation_ratio))
    train_fideids = set(fideids[:split_index])
    validation_split_index = split_index + int(len(fideids) * validation_ratio)
    validation_fideids = set(fideids[split_index:validation_split_index])
    test_fideids = set(fideids[validation_split_index:])
    # TODO: MAke it games over year.

    return df[df["fideid"].isin(train_fideids)].reset_index(drop=True), df[df["fideid"].isin(validation_fideids)].reset_index(drop=True), df[df["fideid"].isin(test_fideids)].reset_index(drop=True)

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out.squeeze()

def extract_linear_model_data(df: pd.DataFrame, lag):
    # pooled regression with arima errors. Look up python or r code.
    X_columns = ["time", "age_at_time", "elo", "games"]

    player_Xs = []
    player_fide_ids = []
    player_Ys = []

    for fide_id, player_df in df.groupby("fideid"):
        player_df = player_df.copy()
        player_df["next_year_elo"] = player_df["elo"].shift(-lag)
        player_df = player_df.dropna(subset=["next_year_elo"])
        
        player_Xs.append(player_df[X_columns])
        player_Ys.append(player_df["next_year_elo"].values)
        player_fide_ids += [fide_id] * len(player_df)

    return pd.concat(player_Xs, ignore_index=True), np.hstack(player_Ys), player_fide_ids

def train_elo_prediction_linear_model(timeseries_df: pd.DataFrame, lag=1):
    # First we split and scale our data
    data_splitter_and_scaler = DataSplitterAndScaler(timeseries_df, test_ratio=0.2, validation_ratio=0)

    X_train, y_train, player_fide_ids = extract_linear_model_data(data_splitter_and_scaler.train_df, lag)
    X_test, y_test, player_fide_ids = extract_linear_model_data(data_splitter_and_scaler.test_df, lag)

    model = LinearRegression().fit(X_train, y_train)
    display = PredictionErrorDisplay.from_estimator(
        model, X_train, y_train, kind="residual_vs_predicted"
    )

    # Let's plot everything, so we can tell what's going on and report on it later.
    # This is so heteroskedastic that it's not even funny.
    display.plot()
    plt.title(f"Residuals vs Predicted Values for Linear Regression Model, lag = {lag}")
    plt.savefig(LINEAR_MODEL_PLOT_DIR / f"residuals_vs_predicted_lag_{lag}.png")
    plt.clf()
    plt.cla()
    plt.close()

    differences = model.predict(X_train) - y_train
    differences_by_fideid = {}
    for fide_id, difference in zip(player_fide_ids, differences):
        if fide_id not in differences_by_fideid:
            differences_by_fideid[fide_id] = []
        differences_by_fideid[fide_id].append(difference)
    
    for fide_id, differences in differences_by_fideid.items():
        plt.plot(differences, label=fide_id)
    plt.title(f"Residuals over Time for Each Player, lag = {lag}")
    plt.xlabel("Time Step")
    plt.ylabel("Residual")
    plt.savefig(LINEAR_MODEL_PLOT_DIR / f"residuals_over_time_by_player_lag_{lag}.png")
    plt.clf()
    plt.cla()
    plt.close()

    # Finally, let's look at the autocorrelation of our residual sequences, to see if ARIMAX might be a good idea.
    lags = []
    autocorrelations = []
    for fide_id, differences in differences_by_fideid.items():
        autocorr = acf(differences, fft=True)
        
        for i in range(1, len(autocorr)):
            lags.append(i)
            autocorrelations.append(autocorr[i]/autocorr[0])
    plt.hist2d(lags, autocorrelations, bins=30, cmap='Blues')
    plt.title(f"Residual autocorrelations, scatter. lag = {lag}")
    plt.xlabel("Lag")
    plt.ylabel("Autcorrelation")
    plt.savefig(LINEAR_MODEL_PLOT_DIR / f"autocorrelations_for_linear_residuals_{lag}.png")
    plt.clf()
    plt.cla()
    plt.close()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lags = []
        partial_autocorrelations = []
        for fide_id, differences in differences_by_fideid.items():
            if len(differences) < 2:
                continue

            partial_autocorr = pacf(differences)
            
            for i in range(1, len(partial_autocorr)):
                lags.append(i)
                partial_autocorrelations.append(partial_autocorr[i])
    plt.hist2d(lags, partial_autocorrelations, bins=30, cmap='Blues', range=[[1, max(lags)], [-1, 1]])
    plt.title(f"Residual partial autocorrelations, scatter. lag = {lag}")
    plt.xlabel("Lag")
    plt.ylabel("Partial Autcorrelation")
    plt.savefig(LINEAR_MODEL_PLOT_DIR / f"partial_autocorrelations_for_linear_residuals_{lag}.png")
    plt.clf()
    plt.cla()
    plt.close()


    # Let's return our RMSE on the test set
    return np.sqrt(mean_squared_error(y_test, model.predict(X_test))) * data_splitter_and_scaler.get_elo_scale()

def exponential_smoothing(timeseries_df: pd.DataFrame, lag=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # First we split and scale our data
        data_splitter_and_scaler = DataSplitterAndScaler(timeseries_df, test_ratio=0.2, validation_ratio=0)

        # We're going to try several values of alpha on the training set, choose the one that minimizes SSE
        # Then return the result that gives on the test set.
        candidate_alphas = np.linspace(0.01, 1, 10)
        best_alpha = None
        best_sse = float("inf")

        for alpha in candidate_alphas:
            total_sse = 0
            for fide_id, player_df in data_splitter_and_scaler.train_df.groupby("fideid"):
                player_df = player_df.copy()
                if len(player_df["elo"]) < 2:
                    continue
                model = SimpleExpSmoothing(player_df["elo"]).fit(smoothing_level=alpha, optimized=False)
                player_df["predicted_elo"] = model.fittedvalues
                player_df["predicted_elo"] = player_df["predicted_elo"].shift(lag)
                player_df = player_df.dropna(subset=["predicted_elo"])
                player_sse = ((player_df["elo"] - player_df["predicted_elo"]) ** 2).sum()
                total_sse += player_sse
            
            if total_sse < best_sse:
                best_sse = total_sse
                best_alpha = alpha
        
        # Now, let's do the same thing on the test set, but with our best alpha.
        square_errors = []
        for fide_id, player_df in data_splitter_and_scaler.test_df.groupby("fideid"):
            player_df = player_df.copy()
            model = SimpleExpSmoothing(player_df["elo"]).fit(smoothing_level=best_alpha, optimized=False)
            player_df["predicted_elo"] = model.fittedvalues
            player_df["predicted_elo"] = player_df["predicted_elo"].shift(lag)
            player_df = player_df.dropna(subset=["predicted_elo"])
            square_errors.extend((player_df["elo"] - player_df["predicted_elo"]) ** 2)
        
        return np.sqrt(np.mean(square_errors)) * data_splitter_and_scaler.get_elo_scale()

class DataSplitterAndScaler:
    def __init__(self, df: pd.DataFrame, test_ratio=0.2, validation_ratio=0.2):
        new_df = df.copy()
        new_df["time"] = (new_df["time"] - pd.Timestamp("1968-01-01")) // pd.Timedelta('1s')    # A bit older than unix time, since the first elo annoyingly comes out right before it.
        self.train_df, self.validation_df, self.test_df = split_timeseries_data_table_into_train_and_test(new_df, test_ratio, validation_ratio)

        self.scaler = MinMaxScaler()
        self.train_df[self.train_df.columns.difference(["fideid"])] = self.scaler.fit_transform(self.train_df[self.train_df.columns.difference(["fideid"])])
        if len(self.validation_df) > 0:
            self.validation_df[self.validation_df.columns.difference(["fideid"])] = self.scaler.transform(self.validation_df[self.validation_df.columns.difference(["fideid"])])
        self.test_df[self.test_df.columns.difference(["fideid"])] = self.scaler.transform(self.test_df[self.test_df.columns.difference(["fideid"])])
    
    def get_elo_scale(self):
        feature_names = self.scaler.feature_names_in_
        elo_index = self.scaler.feature_names_in_.tolist().index("elo")
        return self.scaler.data_max_[elo_index] - self.scaler.data_min_[elo_index]

if __name__ == "__main__":
    # Build tables for paper models and print out the first few rows of each model's table
    timeseries_data_table = get_timeseries_data_table()
    print(timeseries_data_table)

    for lag in range(1, 11):
        print(f"Linear Model RMSE (lag={lag}): {train_elo_prediction_linear_model(timeseries_data_table, lag=lag)}")
        print(f"Exponential Smoothing RMSE (lag={lag}): {exponential_smoothing(timeseries_data_table, lag=lag)}")