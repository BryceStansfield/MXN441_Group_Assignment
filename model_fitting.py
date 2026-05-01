import preprocessing
from preprocessing import TabularModelData
import sklearn
from sklearn.linear_model import LinearRegression as SLinearRegression
from sklearn.model_selection import GridSearchCV as SGridSearchCV

# PaperModels
class PaperLinearModel:
    def __init__(self, data: TabularModelData):
        self.data = data
        self.base_model = SLinearRegression()

    def fit(self):
        X = self.data.data_table[self.data.X_columns]
        Y = self.data.data_table[self.data.Y_columns]
        
        self.cv_model = SGridSearchCV(self.base_model, param_grid={}, cv=5, scoring="neg_root_mean_squared_error")  # Here we just use GridSearchCV as a quick and easy way to get CV'd RMSE.
        self.cv_model.fit(X, Y)
    
    def get_cv_rmse(self):
        return -self.cv_model.best_score_

def fit_all_paper_models():
    data = preprocessing.build_tables_for_paper_models(preprocessing.open_filtered_standard_data())

    model_classes = [PaperLinearModel]

    best_models_per_dataset = []

    print("Fitting paper models...")
    for dataset in data:
        best_model_so_far = ""
        best_rmse_so_far = float('inf')

        for model_class in model_classes:
            try:
                model = model_class(dataset)
                model.fit()

                if model.get_cv_rmse() < best_rmse_so_far:
                    best_rmse_so_far = model.get_cv_rmse()
                    best_model_so_far = model_class.__name__
            except Exception as e:
                print(f"Error fitting {model_class.__name__} on dataset {dataset.model_name} with X columns: {dataset.X_columns}, Y columns: {dataset.Y_columns}: {e}")
        
        best_models_per_dataset.append((dataset.model_name, best_model_so_far, best_rmse_so_far))
    print(best_models_per_dataset)

if __name__ == "__main__":
    fit_all_paper_models()