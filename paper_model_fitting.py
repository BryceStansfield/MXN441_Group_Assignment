import pickle

import preprocessing
from preprocessing import TabularModelData
from sklearn.linear_model import LinearRegression as SLinearRegression
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV as SGridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import pathlib
import pandas as pd
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt

HEATMAP_DIR = pathlib.Path("visualizations/heatmaps")
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

def rmses_to_score(rmses):
    rmses = list(rmses)
    if any(map(lambda i: math.isnan(i), rmses)):
        return float('inf')
    return -sum(rmses)/len(rmses)

# PaperModels, cv n=5 like in the paper.
class PaperLinearModel:
    def __init__(self, data: TabularModelData):
        self.data = data
        self.base_model = SLinearRegression()

        self.X = self.data.data_table[self.data.X_columns]
        self.Y = self.data.data_table[self.data.Y_columns].values.ravel()
        self.rmses = None
        self.cv_model = self.base_model
    
    def get_full_model_fit(self):
        self.cv_model.fit(self.X, self.Y)
        return self.cv_model

    def get_cv_rmse(self):
        if self.rmses is None:
            self.rmses = cross_val_score(self.cv_model, self.X, self.Y, scoring="neg_root_mean_squared_error", cv=min(10, len(self.Y)))
        return rmses_to_score(self.rmses)    
    def get_best_params(self):
        raise NotImplementedError()

class PaperGradientBoostingModel:
    def __init__(self, data: TabularModelData):
        self.data = data
        self.base_model = xgb.XGBRegressor(random_state=1)

        self.X = self.data.data_table[self.data.X_columns]
        self.Y = self.data.data_table[self.data.Y_columns].values.ravel()
        self.rmses = None
        
        self.cv_model = SGridSearchCV(self.base_model,
                                      param_grid={"n_estimators": [10, 20, 50, 100, 200, 500, 1000],
                                                  "max_depth": [1, 3, 5],
                                                  "learning_rate": [0.1, 0.2, 0.3]},
                                      cv=5,
                                      scoring="neg_root_mean_squared_error",
                                      n_jobs=-1)
    
    def get_full_model_fit(self):
        self.cv_model.fit(self.X, self.Y)
        return self.cv_model

    def get_cv_rmse(self):
        if self.rmses is None:
            self.rmses = cross_val_score(self.cv_model, self.X, self.Y, scoring="neg_root_mean_squared_error", cv=min(10, len(self.Y)))
        return rmses_to_score(self.rmses)

    def get_best_params(self):
        raise NotImplementedError()

class PaperAdaBoostModel:
    def __init__(self, data: TabularModelData):
        self.data = data
        self.base_model = AdaBoostRegressor(random_state=1)
        self.X = self.data.data_table[self.data.X_columns]
        self.Y = self.data.data_table[self.data.Y_columns].values.ravel()
        self.rmses = None
        
        self.cv_model = SGridSearchCV(self.base_model,
                                      param_grid={"n_estimators": [10, 20, 50, 100, 200, 500, 1000],
                                                  "learning_rate": [0.2, 0.5, 1, 2, 5]},
                                      cv=5,
                                      scoring="neg_root_mean_squared_error",
                                      n_jobs=-1)
        
    
    def get_full_model_fit(self):
        self.cv_model.fit(self.X, self.Y)
        return self.cv_model

    def get_cv_rmse(self):
        if self.rmses is None:
            self.rmses = cross_val_score(self.cv_model, self.X, self.Y, scoring="neg_root_mean_squared_error", cv=min(10, len(self.Y)))
        return rmses_to_score(self.rmses)
    def get_best_params(self):
        raise NotImplementedError()


class PaperMLPModel:
    def __init__(self, data: TabularModelData):
        self.data = data
        self.base_model = MLPRegressor(max_iter = 10000, random_state=1)
        self.X = self.data.data_table[self.data.X_columns]
        self.Y = self.data.data_table[self.data.Y_columns].values.ravel()
        self.rmses = None
        
        self.cv_model = SGridSearchCV(self.base_model,
                                      param_grid={"hidden_layer_sizes": [(10,), (50,), (100,), (10, 10), (50, 50), (100, 100), (10, 10, 10), (50, 50, 50), (100, 100, 100)],
                                                  "learning_rate_init": [0.001, 0.01, 0.1],
                                                  "alpha": [0.0001, 0.001, 0.01]},
                                      cv=5,
                                      scoring="neg_root_mean_squared_error",
                                      n_jobs=-1)

    def get_full_model_fit(self):
        self.cv_model.fit(self.X, self.Y)
        return self.cv_model

    def get_cv_rmse(self):
        if self.rmses is None:
            self.rmses = cross_val_score(self.cv_model, self.X, self.Y, scoring="neg_root_mean_squared_error", cv=min(10, len(self.Y)))
        return rmses_to_score(self.rmses)
    def get_best_params(self):
        raise NotImplementedError()


class PaperElasticModel:
    # Note that the paper l1 model is a subset of this. l2 isn't, since this isn't numerically stable for l1_ratio = 0.
    def __init__(self, data: TabularModelData):
        self.data = data
        self.base_model = ElasticNet(max_iter=10000, random_state=1)
        self.X = self.data.data_table[self.data.X_columns]
        self.Y = self.data.data_table[self.data.Y_columns].values.ravel()
        self.rmses = None

        self.cv_model = SGridSearchCV(self.base_model,
                                      param_grid={"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                                                  "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]},
                                      cv=5,
                                      scoring="neg_root_mean_squared_error",
                                      n_jobs=-1)

    def get_full_model_fit(self):
        self.cv_model.fit(self.X, self.Y)
        return self.cv_model

    def get_cv_rmse(self):
        if self.rmses is None:
            self.rmses = cross_val_score(self.cv_model, self.X, self.Y, scoring="neg_root_mean_squared_error", cv=min(10, len(self.Y)))
        return rmses_to_score(self.rmses)    
    def get_best_params(self):
        raise NotImplementedError()


class PaperRidgeModel:
    def __init__(self, data: TabularModelData):
        self.data = data
        self.base_model = Ridge(max_iter=10000, random_state=1)

        self.X = self.data.data_table[self.data.X_columns]
        self.Y = self.data.data_table[self.data.Y_columns].values.ravel()
        self.rmses = None

        self.cv_model = SGridSearchCV(self.base_model,
                                      param_grid={"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]},
                                      cv=5,
                                      scoring="neg_root_mean_squared_error",
                                      n_jobs=-1)

    def get_full_model_fit(self):
        self.cv_model.fit(self.X, self.Y)
        return self.cv_model

    def get_cv_rmse(self):
        if self.rmses is None:
            self.rmses = cross_val_score(self.cv_model, self.X, self.Y, scoring="neg_root_mean_squared_error", cv=min(10, len(self.Y)))
        return rmses_to_score(self.rmses)    
    def get_best_params(self):
        raise NotImplementedError()


class PaperkNNModel:
    def __init__(self, data: TabularModelData):
        self.data = data
        self.base_model = KNeighborsRegressor()
        self.X = self.data.data_table[self.data.X_columns]
        self.Y = self.data.data_table[self.data.Y_columns].values.ravel()
        self.rmses = None
        
        self.cv_model = SGridSearchCV(self.base_model,
                                      param_grid={"n_neighbors": [1, 3, 5, 10, 20, 50, 100]},
                                      cv=5,
                                      scoring="neg_root_mean_squared_error",
                                      n_jobs=-1)

    def get_full_model_fit(self):
        self.cv_model.fit(self.X, self.Y)
        return self.cv_model

    def get_cv_rmse(self):
        if self.rmses is None:
            self.rmses = cross_val_score(self.cv_model, self.X, self.Y, scoring="neg_root_mean_squared_error", cv=min(10, len(self.Y)))
        return rmses_to_score(self.rmses)    
    def get_best_params(self):
        raise NotImplementedError()

class PaperSVRModel:
    def __init__(self, data: TabularModelData):
        self.data = data
        self.base_model = SVR()
        self.X = self.data.data_table[self.data.X_columns]
        self.Y = self.data.data_table[self.data.Y_columns].values.ravel()
        self.rmses = None
        
        self.cv_model = SGridSearchCV(self.base_model,
                                      param_grid={"C": [0.1, 1, 10, 100],
                                                  "gamma": [0.001, 0.01, 0.1, 1],
                                                  "kernel": ["rbf", "linear"]},
                                      cv=5,
                                      scoring="neg_root_mean_squared_error",
                                      n_jobs=-1)

    def get_full_model_fit(self):
        self.cv_model.fit(self.X, self.Y)
        return self.cv_model

    def get_cv_rmse(self):
        if self.rmses is None:
            self.rmses = cross_val_score(self.cv_model, self.X, self.Y, scoring="neg_root_mean_squared_error", cv=min(10, len(self.Y)))
        return rmses_to_score(self.rmses)    
    def get_best_params(self):
        raise NotImplementedError()

class PaperRFModel:
    def __init__(self, data: TabularModelData):
        self.data = data
        self.base_model = RandomForestRegressor(random_state=1)
        self.X = self.data.data_table[self.data.X_columns]
        self.Y = self.data.data_table[self.data.Y_columns].values.ravel()
        self.rmses = None
        
        self.cv_model = SGridSearchCV(self.base_model,
                                      param_grid={"n_estimators": [10, 20, 50, 100, 200, 500, 1000],
                                                  "max_depth": [1, 3, 5]},
                                      cv=5,
                                      scoring="neg_root_mean_squared_error",
                                      n_jobs=-1)
    
    def get_full_model_fit(self):
        self.cv_model.fit(self.X, self.Y)
        return self.cv_model

    def get_cv_rmse(self):
        if self.rmses is None:
            self.rmses = cross_val_score(self.cv_model, self.X, self.Y, scoring="neg_root_mean_squared_error", cv=min(10, len(self.Y)))
        return rmses_to_score(self.rmses)

    def get_best_params(self):
        raise NotImplementedError()

    
class PaperDTModel:
    def __init__(self, data: TabularModelData):
        self.data = data
        self.base_model = DecisionTreeRegressor(random_state=1)

        self.X = self.data.data_table[self.data.X_columns]
        self.Y = self.data.data_table[self.data.Y_columns].values.ravel()
        self.rmses = None
        
        self.cv_model = SGridSearchCV(self.base_model,
                                      param_grid={"max_depth": [1, 3, 5, 10, 20, 50, 100]},
                                      cv=5,
                                      scoring="neg_root_mean_squared_error",
                                      n_jobs=-1)

    def get_full_model_fit(self):
        self.cv_model.fit(self.X, self.Y)
        return self.cv_model

    def get_cv_rmse(self):
        if self.rmses is None:
            self.rmses = cross_val_score(self.cv_model, self.X, self.Y, scoring="neg_root_mean_squared_error", cv=min(10, len(self.Y)))
        return rmses_to_score(self.rmses)
    
    def get_best_params(self):
        raise NotImplementedError()

def get_X_ranges(data: TabularModelData):
    col_ranges = {}

    for col in data.X_columns:
        min = data.data_table[col].min()
        max = data.data_table[col].max()

        col_ranges[col] = {"min": min, "max": max}
    return col_ranges

def model_heatmap(data: TabularModelData, model, steps=30):
    fit_model = model.get_full_model_fit()
    X_ranges = get_X_ranges(data)
    X_col_names = [x for x in X_ranges]
    if len(X_col_names) > 2:
        return # No 3d heatmaps.
    
    X_steps = [list(np.linspace(X_ranges[x]["min"], X_ranges[x]["max"], num=steps)) for x in X_ranges]

    X = []
    for X_s in itertools.product(*X_steps):
        d = {}
        for i in range(len(X_s)):
            d[X_col_names[i]] = X_s[i]
        X.append(d)
    
    X_df = pd.DataFrame(X)
    Ys = fit_model.predict(X_df)

    if len(X_col_names) == 2:
        plt.clf()
        IM = [[0 for i in range(steps)] for j in range(steps)]
        for i, (x, y) in enumerate(itertools.product([i for i in range(steps)], [i for i in range(steps)])):
            IM[y][x] = Ys[i]
        plt.imshow(IM)
        plt.colorbar()
        plt.xlabel(X_col_names[0])
        plt.xticks([0, steps-1], [round(X_steps[0][0]), round(X_steps[0][-1])])
        plt.ylabel(X_col_names[1])
        plt.yticks([0, steps-1], [round(X_steps[1][0]), round(X_steps[1][-1])])
        plt.title(data.Y_columns[0])
        plt.savefig(HEATMAP_DIR / f"{data.model_name}_best_model_heatmap.png")
    if len(X_col_names) == 1:
        plt.clf()
        IM = np.array(X_df[X_col_names[0]])
        IM = IM.reshape(-1, 1)
        plt.imshow(IM)
        plt.colorbar()
        plt.ylabel(X_col_names[0])
        plt.yticks([0, steps-1], [round(X_steps[0][0]), round(X_steps[0][-1])])
        plt.xticks([0], [""])
        plt.title(data.Y_columns[0])
        plt.savefig(HEATMAP_DIR / f"{data.model_name}_best_model_heatmap.png")
        return

def fit_all_paper_models(use_cache = True, verbose = True):
    data = preprocessing.build_tables_for_paper_models(preprocessing.open_filtered_standard_data())

    model_classes = [PaperLinearModel, PaperGradientBoostingModel, PaperAdaBoostModel, PaperMLPModel, PaperElasticModel, 
                     PaperRidgeModel, PaperkNNModel, PaperSVRModel, PaperRFModel, PaperDTModel]

    best_models_per_dataset = []
    model_performances = []

    # Cache directory
    cache_directory = pathlib.Path("cache/paper_models")
    cache_directory.mkdir(parents=True, exist_ok=True)

    print("Fitting paper models...")
    for dataset in data:
        best_model_name_so_far = ""
        best_rmse_so_far = float('inf')

        for model_class in model_classes:
            try:
                model_filepath = cache_directory / f"{dataset.model_name}_{model_class.__name__}.pkl"
                if use_cache and model_filepath.exists():
                    if verbose:
                        print(f"Loading cached model for {model_class.__name__} on dataset {dataset.model_name} from {model_filepath}...")
                    
                    with open(model_filepath, "rb") as f:
                        model = pickle.load(f)
                    
                    if verbose:
                        print(f"Finished loading cached model for {model_class.__name__} on dataset {dataset.model_name}. CV RMSE: {model.get_cv_rmse():.4f}")
                else:
                    model = model_class(dataset)
                    if verbose:
                        print(f"Fitting {model_class.__name__} on dataset {dataset.model_name} with X columns: {dataset.X_columns}, Y columns: {dataset.Y_columns}...")
                    
                    model.get_cv_rmse()
                    if use_cache:
                        with open(model_filepath, "wb") as f:
                            pickle.dump(model, f)

                    if verbose:
                        print(f"Finished fitting {model_class.__name__} on dataset {dataset.model_name}. CV RMSE: {model.get_cv_rmse():.4f}")

                model_rmse = model.get_cv_rmse()

                model_performances.append({"ml_model": model_class.__name__, "paper_model":dataset.model_name, "rmse":model_rmse})

                if model.get_cv_rmse() < best_rmse_so_far:
                    best_rmse_so_far = model.get_cv_rmse()
                    best_model_name_so_far = model_class.__name__
                    best_model_so_far = model

                    if verbose:
                        print(f"New best model for dataset {dataset.model_name}: {best_model_name_so_far} with CV RMSE: {best_rmse_so_far:.4f}")
            except Exception as e:
                raise e
        
        best_models_per_dataset.append((dataset.model_name, best_model_name_so_far, best_rmse_so_far))
        model_heatmap(dataset, best_model_so_far)
    
    for i in range(len(best_models_per_dataset)):
        model_name, best_model, best_rmse = best_models_per_dataset[i]
        print(f"Best model for dataset {model_name} (n={len(data[i].data_table)}): {best_model} with CV RMSE: {best_rmse:.4f}")
    
    model_performances_df = pd.DataFrame(model_performances)
    model_performances_df.to_csv("cache/paper_model_performances.csv")
    print(model_performances_df.to_string())

if __name__ == "__main__":
    fit_all_paper_models()