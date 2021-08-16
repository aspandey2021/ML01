import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def featureEngineering(df):
    """
    performs feature engineering like one-hot encoding, log transformation on the dataframe
    :param df: dataframe
    :return: the modified dataframe
    """
    #One-hot encoding the categorical features
    categ = df.select_dtypes('object').columns.to_list()
    num_ohc_cols = (df[categ].apply(lambda x: x.nunique()).sort_values(ascending=False))
    ohc = OneHotEncoder()

    for col in num_ohc_cols.index:
        # One hot encode this data - this returns a sparse array
        new_dat = ohc.fit_transform(df[[col]])

        # drop original column from Dataframe
        df = df.drop(col, axis=1)

        # get names of all unique elements so that one can identify them later
        cats = ohc.categories_

        # Create column names for each OHE column by value
        new_cols = ['_'.join([col, cat]) for cat in cats[0]]

        # create the new dataframe
        new_df = pd.DataFrame(new_dat.toarray(), columns=new_cols)

        # Append the new data to the dataframe
        df = pd.concat([df, new_df], axis=1)

    #Applying Log transformation to skewed feature 'Item_Visibility'
    df['Item_Visibility'] = df['Item_Visibility'].apply(np.sqrt)

    return df
