import torch

import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot

from preprocessing import get_full_timeseries_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import PredictionErrorDisplay
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
import pathlib
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.preprocessing import StandardScaler

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics.point import RMSE
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.data.encoders import NaNLabelEncoder

import pickle

import warnings
warnings.filterwarnings("ignore")   # Pytorch forecasting throws an inordinate number of warnings

import logging

class TipFilter(logging.Filter):
    def filter(self, record):
        return "💡 Tip" not in record.getMessage()

logging.getLogger('lightning.pytorch.utilities.rank_zero').addFilter(TipFilter())

ELO_CUTOFF = 2500
TRANSFORMER_CACHE_DIRECTORY = pathlib.Path("cache") / f"transformer_models_{ELO_CUTOFF}"
TRANSFORMER_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)

PLOT_DIR = pathlib.Path("plots")
LINEAR_MODEL_PLOT_DIR = PLOT_DIR / f"linear_models_{ELO_CUTOFF}"
LINEAR_MODEL_PLOT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PLOT_DIR = PLOT_DIR / "timeseries_result_plots"
RESULT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

def plot_model_predictions_vs_reality(reality, predictions, model_name, sample_name, lag):
    plt.clf()
    x_axis = [i for i in range(len(reality))]
    plt.plot(x_axis, reality)
    plt.plot(x_axis, predictions)
    plt.ylabel("Elo")
    plt.title(f"{model_name} (lag {lag}), {sample_name}")
    plt.savefig(RESULT_PLOT_DIR / f"{model_name}_{lag}_{sample_name}.png")


# Data preprocessing
def yearly_first_data_subset(full_df: pd.DataFrame):
    games_map = {}
    year_time_groupby = full_df.groupby(["fideid", full_df["time"].dt.year])

    for (id, year), df in year_time_groupby:
        df.sort_values(["time"])
        for i, row in enumerate(df.itertuples()):
            if i == 0:
                if (id, year) in games_map:
                    games_map[(id, year)] += row.games
                else:
                    games_map[(id, year)] = row.games
            else:
                if (id, year+1) in games_map:
                    games_map[(id, year+1)] += row.games
                else:
                    games_map[(id, year+1)] = row.games
    
    first_yearly_point = full_df.loc[year_time_groupby["time"].idxmin()].sort_values(["fideid", "time"]).reset_index(drop=True)
    first_yearly_point = first_yearly_point.copy()
    first_yearly_point["games"] = first_yearly_point.apply(lambda row: games_map[(row['fideid'], row['time'].year)], axis=1)

    return first_yearly_point

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

    return df[df["fideid"].isin(train_fideids)].reset_index(drop=True), df[df["fideid"].isin(validation_fideids)].reset_index(drop=True), df[df["fideid"].isin(test_fideids)].reset_index(drop=True)

def plot_model_predictions(ys_preds, elo_scale, elo_minimum,  output_name):
    output = ys_preds[1]
    ys = ys_preds[0]
    plt.clf()
    plt.plot([i for i in range(len(ys))], ys * elo_scale + elo_minimum, label="Real Values")
    plt.plot([i  for i in range(len(output))], output * elo_scale + elo_minimum, label="Predictions")
    plt.legend()
    plt.savefig(RESULT_PLOT_DIR / output_name)

def extract_linear_model_data(df: pd.DataFrame, lag, scaler=None):
    # pooled regression with arima errors. Look up python or r code.
    X_columns = ["time", "age_at_time", "elo", "games", "age*games"]

    player_Xs = []
    player_fide_ids = []
    player_Ys = []

    for fide_id, player_df in df.groupby("fideid"):
        player_df = player_df.copy()
        player_df["age*games"] = player_df["age_at_time"] * player_df["games"]
        player_df["next_year_elo"] = player_df["elo"].shift(-lag)
        player_df = player_df.dropna(subset=["next_year_elo"])
        
        player_Xs.append(player_df[X_columns])
        player_Ys.append(player_df["next_year_elo"].values)
        player_fide_ids += [fide_id] * len(player_df)
    
    player_Xs = pd.concat(player_Xs, ignore_index=True)
    if scaler is None:
        scaler = StandardScaler()
        player_Xs = scaler.fit_transform(player_Xs)
    else:
        player_Xs = scaler.transform(player_Xs)

    return player_Xs, np.hstack(player_Ys), player_fide_ids, scaler

def train_transformers(df: pd.DataFrame, max_lag=1, test_run = False):
    df = df.sort_values(["fideid", "time"])
    df["time_idx"] = df.groupby("fideid").cumcount()

    data_splitter_and_scaler = DataSplitterAndScaler(df)

    training = TimeSeriesDataSet(
        data_splitter_and_scaler.train_df,
        time_idx="time_idx",
        group_ids=["fideid"],
        target="elo",
        min_encoder_length=1,
        max_encoder_length=20,
        min_prediction_length=1,
        max_prediction_length=max_lag,
        time_varying_known_reals=["time", "age_at_time"],
        time_varying_unknown_reals=[
            "elo",
            "games"
        ],
        categorical_encoders={"fideid": NaNLabelEncoder(add_nan=True)},
        time_varying_known_categoricals=[],
        time_varying_unknown_categoricals=[],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(
        training, data_splitter_and_scaler.validation_df, predict=False, stop_randomization=True
    )

    test_dataset = TimeSeriesDataSet.from_dataset(
        training, data_splitter_and_scaler.test_df, predict=False, stop_randomization=True
    )

    individual_test_datasets = [
        (test_set, TimeSeriesDataSet.from_dataset(training, test_set, predict=False, stop_randomization=True)) for test_set in data_splitter_and_scaler.split_test_into_individuals(max_lag)
    ]

    # create dataloaders for model
    batch_size = 128
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=10
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=10
    )
    test_dataloader = test_dataset.to_dataloader(
        train=False, batch_size=batch_size, num_workers=10
    )

    # configure network and trainer
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=20, verbose=False, mode="min"
    )

    study_path = TRANSFORMER_CACHE_DIRECTORY / f"transformer_study.pkl"

    print("Doing optuna study for best params")
    if study_path.exists():
        with open(study_path, "rb") as f:
            study = pickle.load(f) 
    else:
        study = optimize_hyperparameters(
            train_dataloader,
            val_dataloader,
            model_path=str(TRANSFORMER_CACHE_DIRECTORY / f"{max_lag}_optuna"),
            n_trials=200,
            max_epochs=50,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(8, 128),
            hidden_continuous_size_range=(8, 128),
            attention_head_size_range=(1, 4),
            learning_rate_range=(0.001, 0.1),
            dropout_range=(0.1, 0.3),
            trainer_kwargs=dict(limit_train_batches=30),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=True
        )

        with open(study_path, "wb") as f:
            pickle.dump(study, f)

    full_model_path = TRANSFORMER_CACHE_DIRECTORY / f"{max_lag}_transformer.model"
    if full_model_path.exists() and not test_run:
        print("Loading Model from cache")

        with open(full_model_path, 'rb') as f:
            kwargs, state_dict = torch.load(full_model_path, weights_only=False)
            tft = TemporalFusionTransformer.from_dataset(
                training,
                loss=RMSE(),
                **kwargs
            )
            tft.load_state_dict(state_dict)
    else:
        print("Training model")
        trainer = pl.Trainer(
            max_epochs=5000 if not test_run else 1,
            accelerator="cpu",
            enable_model_summary=True,
            # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
            callbacks=[early_stop_callback],
            gradient_clip_val=study.best_params["gradient_clip_val"]
        )

        model_params = study.best_params.copy()
        del model_params["gradient_clip_val"]
        tft = TemporalFusionTransformer.from_dataset(
            training,
            loss=RMSE(),
            log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            optimizer="ranger",
            reduce_on_plateau_patience=4,
            **model_params
        )
        print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        if not test_run:
            with open(full_model_path, 'wb') as f:
                torch.save([tft._hparams, tft.state_dict()], f)

    # Let's display our results on our best and worse performing test_set.
    worst_testset_tuples = [None for _ in range(max_lag)]
    worst_testset_mse = [0 for _ in range(max_lag)]
    best_testset_tuples = [None for _ in range(max_lag)]
    best_testset_mse = [10000000000000000 for _ in range(max_lag)]
    all_mses = [[] for _ in range(max_lag)]
    for i, (test_set, test_set_loader) in enumerate(individual_test_datasets):
        predictions = tft.predict(test_set_loader, return_y=True, trainer_kwargs=dict(accelerator="cpu", logger=False))

        for i in range(min(max_lag, predictions.output.size()[1])):
            non_zero_inds = torch.nonzero(predictions.y[0][:, i])
            y_nonzero = predictions.y[0][non_zero_inds, i]
            preds_nonzero = predictions.output[non_zero_inds, i]

            mse = RMSE()(preds_nonzero, y_nonzero) ** 2
            all_mses[i].append((mse, len(y_nonzero),))

            if mse > worst_testset_mse[i]:
                worst_testset_mse[i] = mse
                worst_testset_tuples[i] = (y_nonzero, preds_nonzero)
            if mse < best_testset_mse[i]:
                best_testset_mse[i] = mse
                best_testset_tuples[i] = (y_nonzero, preds_nonzero)
    
    for i in range(max_lag):
        plot_model_predictions(worst_testset_tuples[i], data_splitter_and_scaler.get_elo_scale(), data_splitter_and_scaler.get_elo_minimum(), f"linear_worst_prediction_{i+1}.png")
        plot_model_predictions(best_testset_tuples[i], data_splitter_and_scaler.get_elo_scale(), data_splitter_and_scaler.get_elo_minimum(), f"linear_best_prediction_{i+1}.png")
    
    rmses_by_lag = []
    for i in range(0, max_lag):
        mses = all_mses[i]

        sum_lengths = sum(x[1] for x in mses)
        sum_weighted_mses = sum([x[0] * x[1] for x in mses])
        rmses_by_lag.append((sum_weighted_mses / sum_lengths)**0.5 * data_splitter_and_scaler.get_elo_scale())
    
    return rmses_by_lag
    
def train_full_timeseries_transformer(df: pd.DataFrame):
    df = df.sort_values(["fideid", "time"])
    df["time_idx"] = df.groupby("fideid").cumcount()

    data_splitter_and_scaler = DataSplitterAndScaler(df)

    training = TimeSeriesDataSet(
        data_splitter_and_scaler.train_df,
        time_idx="time_idx",
        group_ids=["fideid"],
        target="elo",
        min_encoder_length=1,
        max_encoder_length=120,
        min_prediction_length=1,
        max_prediction_length=120,
        time_varying_known_reals=["time", "age_at_time"],
        time_varying_unknown_reals=[
            "elo",
            "games"
        ],
        categorical_encoders={"fideid": NaNLabelEncoder(add_nan=True)},
        time_varying_known_categoricals=[],
        time_varying_unknown_categoricals=[],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(
        training, data_splitter_and_scaler.validation_df, predict=False, stop_randomization=True
    )

    test_dataset = TimeSeriesDataSet.from_dataset(
        training, data_splitter_and_scaler.test_df, predict=False, stop_randomization=True
    )

    individual_test_datasets = [
        (test_set, TimeSeriesDataSet.from_dataset(training, test_set, predict=False, stop_randomization=True)) for test_set in data_splitter_and_scaler.split_test_into_individuals()
    ]

    # create dataloaders for model
    batch_size = 128
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=10
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=10
    )
    test_dataloader = test_dataset.to_dataloader(
        train=False, batch_size=batch_size, num_workers=10
    )

    # configure network and trainer
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=2, verbose=False, mode="min"
    )

    study_path = TRANSFORMER_CACHE_DIRECTORY / f"full_transformer_study.pkl"

    print("Doing optuna study for best params")
    if study_path.exists():
        with open(study_path, "rb") as f:
            study = pickle.load(f) 
    else:
        study = optimize_hyperparameters(
            train_dataloader,
            val_dataloader,
            model_path=str(TRANSFORMER_CACHE_DIRECTORY / f"full_optuna"),
            n_trials=200,
            max_epochs=50,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(8, 128),
            hidden_continuous_size_range=(8, 128),
            attention_head_size_range=(1, 4),
            learning_rate_range=(0.001, 0.1),
            dropout_range=(0.1, 0.3),
            trainer_kwargs=dict(limit_train_batches=30),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=True
        )

        with open(study_path, "wb") as f:
            pickle.dump(study, f)

    full_model_path = TRANSFORMER_CACHE_DIRECTORY / f"full_transformer.model"
    loss = QuantileLoss([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    if full_model_path.exists():
        print("Loading Model from cache")

        with open(full_model_path, 'rb') as f:
            kwargs, state_dict = torch.load(full_model_path, weights_only=False)
            tft = TemporalFusionTransformer.from_dataset(
                training,
                loss=loss,
                **kwargs
            )
            tft.load_state_dict(state_dict)
    else:
        print("Training model")
        trainer = pl.Trainer(
            max_epochs=5000,
            accelerator="gpu",
            enable_model_summary=True,
            # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
            callbacks=[early_stop_callback],
            gradient_clip_val=study.best_params["gradient_clip_val"]
        )

        model_params = study.best_params.copy()
        del model_params["gradient_clip_val"]
        tft = TemporalFusionTransformer.from_dataset(
            training,
            loss=loss,
            log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            optimizer="ranger",
            reduce_on_plateau_patience=4,
            **model_params
        )
        print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

        with open(full_model_path, 'wb') as f:
            torch.save([tft._hparams, tft.state_dict()], f)
    
    predictions = tft.predict(
        test_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu")
    )

    return {"rmse": RMSE()(predictions.output, predictions.y) * data_splitter_and_scaler.get_elo_scale()}


def train_elo_prediction_linear_model(timeseries_df: pd.DataFrame, lag=1):
    # First we split and scale our data
    data_splitter_and_scaler = DataSplitterAndScaler(timeseries_df, test_ratio=0.2, validation_ratio=0)

    X_train, y_train, training_player_fide_ids, scaler = extract_linear_model_data(data_splitter_and_scaler.train_df, lag)
    X_test, y_test, test_player_fide_ids, _ = extract_linear_model_data(data_splitter_and_scaler.test_df, lag, scaler)

    model = ElasticNetCV(l1_ratio=[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1.0]).fit(X_train, y_train)
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
    for fide_id, difference in zip(training_player_fide_ids, differences):
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

    # This is incredibly inefficient, but realistically I don't think this code is going to be run by many (any?) other people.
    worst_mse_error = 0
    worst_mse_tuple = None
    best_mse_error = 1000000000
    best_mse_tuple = None

    test_df = pd.DataFrame()    
    test_df["Predicted_Elo"] = model.predict(X_test)
    test_df["Real_Future_Elo"] = y_test
    test_df["fideid"] = test_player_fide_ids

    for fide_id, player_df in test_df.groupby("fideid"):
        player_square_errors = (player_df["Predicted_Elo"].values - player_df["Real_Future_Elo"].values) ** 2
        
        mse = np.mean(player_square_errors)
        if mse > worst_mse_error:
            worst_mse_error = mse
            worst_mse_tuple = (player_df["Real_Future_Elo"].values, player_df["Predicted_Elo"].values,)
        if mse < best_mse_error:
            best_mse_error = mse
            best_mse_tuple = (player_df["Real_Future_Elo"].values, player_df["Predicted_Elo"].values,)
        
    plot_model_predictions(worst_mse_tuple, data_splitter_and_scaler.get_elo_scale(), data_splitter_and_scaler.get_elo_minimum(), f"exp_smoothing_worst_prediction_{lag}.png")
    plot_model_predictions(best_mse_tuple, data_splitter_and_scaler.get_elo_scale(), data_splitter_and_scaler.get_elo_minimum(), f"exp_smoothing_best_prediction_{lag}.png")

    # Let's return our RMSE on the test set
    return {"rmse": np.sqrt(mean_squared_error(test_df["Predicted_Elo"].values, test_df["Real_Future_Elo"].values)) * data_splitter_and_scaler.get_elo_scale()}

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

        best_mse_error = 100000000000
        worst_mse_error = 0
        best_mse_tuple = None
        worst_mse_tuple = None

        for fide_id, player_df in data_splitter_and_scaler.test_df.groupby("fideid"):
            player_df = player_df.copy()
            model = SimpleExpSmoothing(player_df["elo"]).fit(smoothing_level=best_alpha, optimized=False)
            player_df["predicted_elo"] = model.fittedvalues
            player_df["predicted_elo"] = player_df["predicted_elo"].shift(lag)
            player_df = player_df.dropna(subset=["predicted_elo"])
            player_square_errors = (player_df["elo"] - player_df["predicted_elo"]) ** 2
            
            mse = np.mean(player_square_errors)
            if mse > worst_mse_error:
                worst_mse_error = mse
                worst_mse_tuple = (player_df["elo"].values, player_df["predicted_elo"].values,)
            if mse < best_mse_error:
                best_mse_error = mse
                best_mse_tuple = (player_df["elo"].values, player_df["predicted_elo"].values,)
            
            square_errors.extend(player_square_errors)

        plot_model_predictions(worst_mse_tuple, data_splitter_and_scaler.get_elo_scale(), data_splitter_and_scaler.get_elo_minimum(), f"exp_smoothing_worst_prediction_{lag}.png")
        plot_model_predictions(best_mse_tuple, data_splitter_and_scaler.get_elo_scale(), data_splitter_and_scaler.get_elo_minimum(), f"exp_smoothing_best_prediction_{lag}.png")

        return {"rmse": np.sqrt(np.mean(square_errors)) * data_splitter_and_scaler.get_elo_scale()}

class DataSplitterAndScaler:
    def __init__(self, df: pd.DataFrame, test_ratio=0.2, validation_ratio=0.2):
        new_df = df.copy()
        new_df["time"] = (new_df["time"] - pd.Timestamp("1968-01-01")) // pd.Timedelta('1s')    # A bit older than unix time, since the first elo annoyingly comes out right before it.
        self.train_df, self.validation_df, self.test_df = split_timeseries_data_table_into_train_and_test(new_df, test_ratio, validation_ratio)

        self.scaler = MinMaxScaler()
        scaling_cols = self.train_df.columns.difference(["fideid", "time_idx"])
        self.train_df[scaling_cols] = self.scaler.fit_transform(self.train_df[scaling_cols])
        if len(self.validation_df) > 0:
            self.validation_df[scaling_cols] = self.scaler.transform(self.validation_df[scaling_cols])
        self.test_df[scaling_cols] = self.scaler.transform(self.test_df[scaling_cols])

        self.elo_index = self.scaler.feature_names_in_.tolist().index("elo")
    
    def get_elo_scale(self):
        return self.scaler.data_max_[self.elo_index] - self.scaler.data_min_[self.elo_index]
    
    def get_elo_minimum(self):
        return self.scaler.data_min_[self.elo_index]

    def split_test_into_individuals(self, min_length = 5):
        test_sets = []

        for _, df in self.test_df.groupby("fideid"):
            if len(df) >= 1:        # Might change this
                test_sets.append(df)
        
        return test_sets

if __name__ == "__main__":
    full_timeseries_data = get_full_timeseries_model(ELO_CUTOFF)[f"All ever > {ELO_CUTOFF} players"]

    train_full_timeseries_transformer(full_timeseries_data)

    # Build tables for paper models and print out the first few rows of each model's table
    timeseries_data_table = yearly_first_data_subset(full_timeseries_data)
    MAX_LAG = 10

    transformer_rmses = train_transformers(timeseries_data_table, MAX_LAG, test_run=False)

    performances = {"transformer": transformer_rmses, "pooled_linear": [], "simple_exponential_smoothing": []}

    for lag in range(1, MAX_LAG+1):
        performances["pooled_linear"].append(train_elo_prediction_linear_model(timeseries_data_table, lag=lag)["rmse"])
        performances["simple_exponential_smoothing"].append(exponential_smoothing(timeseries_data_table, lag=lag)["rmse"])

    print(performances)
    
    # Lets plot our performances.
    plt.clf()
    
    for key in performances:
        plt.plot([i for i in range(1, MAX_LAG+1)], list(float(x) for x in performances[key]), label=key)
    plt.xlabel("Lag")
    plt.ylabel("rmse (elo)")
    plt.title("Model performances vs lag")
    plt.legend()
    plt.savefig(PLOT_DIR / f"{MAX_LAG}_timeseries_model_performances.png")
    
    pd.DataFrame(performances).to_csv("cache/timeseries_performances.csv")