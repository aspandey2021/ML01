import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold, GridSearchCV,train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def regression_training(df):
    """
    takes Random Forest model with the best parameters
    and trains on the whole data set and saves the model in a pickle file
    :param df: dataframe ready to be trained upon
    :return: the trained model
    """
    # Storing all columns except the target column in a variable
    feature_cols = [x for x in df.columns if x != 'Item_Outlet_Sales']
    # Split the data into train and test
    x_train, x_test, y_train, y_test = train_test_split(df[feature_cols],
                                                        df['Item_Outlet_Sales'],
                                                        test_size=0.3,
                                                        random_state=42)
    kf = KFold(shuffle=True, random_state=42, n_splits=5)

    # Defining and training the Model
    model_name = "Random Forest"
    param_grid = {'max_depth': [5],
                  'max_features': ['auto'],
                  'n_estimators': [100],
                  'criterion': ['mse']}
    model = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, n_jobs=-1, cv=kf)
    model.fit(x_train,y_train)
    print("The best score for {} model is {:.4f}".format(model_name, model.best_score_))
    print("The best parameters for {} model are {}".format(model_name, model.best_params_))
    print("\n")

    #Prediction
    ypred = model.predict(x_test)
    r2 = r2_score(y_test, ypred)
    rmse_m = rmse(y_test, ypred)
    print(" The R2 score for {} model is {:.4f}".format(model_name, r2))
    print(" The RMSE for {} model is {:.4f}".format(model_name, rmse_m))
    print("\n")

    # Feature Importances
    feature_importances = pd.DataFrame(model.best_estimator_.feature_importances_,
                                       index=x_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    print(" Top 5 feature importances:\n",feature_importances.head(5))

    print(" Training completed")

    print("\nSaving File to disk...")
    with open("bigmart_sales_trained_model.pickle", 'wb') as fp:
        pickle.dump(model, fp)

    print("\nProcess Finished")


def rmse(ytrue, ypredicted):
    return np.sqrt(mean_squared_error(ytrue, ypredicted))