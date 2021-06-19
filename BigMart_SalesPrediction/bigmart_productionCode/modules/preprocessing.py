
def preprocess(df):
    """
    cleans the raw data
    :param df: dataframe
    :return: the cleaned dataframe
    """
    print("Started preprocessing ...\n")

    # Replace the missing values or null values
    df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].dropna().mode().values[0])
    df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].dropna().mean())
    median_visib = df['Item_Visibility'][df['Item_Visibility'] > 0.00000].median()
    df.loc[df['Item_Visibility'] == 0.0, 'Item_Visibility'] = median_visib

    return df
