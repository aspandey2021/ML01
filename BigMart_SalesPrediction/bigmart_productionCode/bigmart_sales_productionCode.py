"""
Author : Ashish Pandey

This script trains a model to read in a Bigmart sales csv file,
preprocess, train over 70% of the data, test on the rest, output the
metrics and save the model in a pickle file.
"""
from datetime import datetime
import pandas as pd
from modules.preprocessing import *
from modules.eda import *
from modules.feat_engineering import *
from modules.regression_training import *

class BigmartSales():
    print("Reading in the data file...")
    df = pd.read_csv("Sales_data.csv")

    df_pre = preProcess(df)
    df_eda = exploratoryAnalysis(df_pre)
    df_feat = featureEngineering(df_eda)
    regressionTraining(df_feat)


if __name__ == "__main__":
    start_time = datetime.now()
    BigmartSales()
    print("___ Script took {} seconds ___".format(datetime.now() - start_time))
