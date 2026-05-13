import torch

from preprocessing import build_tables_for_paper_models, get_full_timeseries_model, open_filtered_standard_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn.utils.rnn as rnn_utils
import pathlib
import statsmodels.api as sm

# Data preprocessing
def yearly_first_data_subset(df: pd.DataFrame):
    return df.loc[df.groupby(["fideid", df["time"].dt.year])["time"].idxmin()].sort_values(["fideid", "time"]).reset_index(drop=True)

def get_timeseries_data_table(elo_threshold=2700):
    return yearly_first_data_subset(get_full_timeseries_model(elo_threshold)["All ever > 2700 players"])

def split_timeseries_data_table_into_train_and_test(df: pd.DataFrame, test_ratio=0.2, validation_ratio = 0.2):
    fideids = df["fideid"].unique()
    np.random.Generator.shuffle(np.random.default_rng(42), fideids)

    split_index = int(len(fideids) * (1 - test_ratio - validation_ratio))
    train_fideids = set(fideids[:split_index])
    validation_split_index = split_index + int(len(fideids) * validation_ratio)
    validation_fideids = set(fideids[split_index:validation_split_index])
    test_fideids = set(fideids[validation_split_index:])

    return df[df["fideid"].isin(train_fideids)].reset_index(drop=True), df[df["fideid"].isin(validation_fideids)].reset_index(drop=True), df[df["fideid"].isin(test_fideids)].reset_index(drop=True)

def turn_df_into_lstm_windows(df: pd.DataFrame):
    Xs = []
    X_lengths = []
    y = []

    for fideid, player_df in df.groupby("fideid"):
        for i in range(len(player_df)):
            Xs.append(player_df.iloc[0:i+1].drop(columns=["elo"]).values)
            #X_lengths.append(i+1)
            y.append(player_df.iloc[i]["elo"])

    padded_Xs = rnn_utils.pad_sequence([torch.tensor(x) for x in Xs], batch_first=True)
    
    return rnn_utils.pack_padded_sequence(padded_Xs, X_lengths), np.array(y)

def extract_linear_model_data(df: pd.DataFrame):
    X_columns = ["time", "age_at_time", "elo", "games"]

    player_Xs = []
    player_Ys = []

    for _, player_df in df.groupby("fideid"):
        player_df["next_year_elo"] = player_df["elo"].shift(-1)
        player_df = player_df.dropna(subset=["next_year_elo"])
        
        player_Xs.append(player_df[X_columns])
        player_Ys.append(player_df["next_year_elo"].values)

    return pd.concat(player_Xs), np.hstack(player_Ys)

class DataSplitterAndScaler:
    def __init__(self, df: pd.DataFrame, test_ratio=0.2, validation_ratio=0.2):
        new_df = df.copy()
        new_df["time"] = new_df["time"].view('int64')
        self.train_df, self.validation_df, self.test_df = split_timeseries_data_table_into_train_and_test(new_df, test_ratio, validation_ratio)

        self.scaler = MinMaxScaler()
        self.train_df[self.train_df.columns.difference(["fideid"])] = self.scaler.fit_transform(self.train_df[self.train_df.columns.difference(["fideid"])])
        self.validation_df[self.validation_df.columns.difference(["fideid"])] = self.scaler.transform(self.validation_df[self.validation_df.columns.difference(["fideid"])])
        self.test_df[self.test_df.columns.difference(["fideid"])] = self.scaler.transform(self.test_df[self.test_df.columns.difference(["fideid"])])

if __name__ == "__main__":
    # Build tables for paper models and print out the first few rows of each model's table
    print(extract_linear_model_data(get_timeseries_data_table()))
    #turn_df_into_lstm_windows(get_timeseries_data_table())