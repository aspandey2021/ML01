
def exploratory_analysis(df):
    """
    performs exploratory data analysis
    :param df: dataframe
    :return: the dataframe
    """
    # applying uniform names to the item fat content categories
    df['Item_Fat_Content'].replace(['LF', 'reg', 'low fat'], ['Low Fat', 'Regular', 'Low Fat'], inplace=True)

    # creating a new feature based on the store age rather than the establishment year
    df['Outlet_Age'] = df['Outlet_Establishment_Year'].apply(lambda x: 2021 - x)
    df.drop('Outlet_Establishment_Year', axis=1, inplace=True)

    # dropping non-numerical Identifier columns which do not provide any usage for regression analysis
    df.drop('Item_Identifier', axis=1, inplace=True)
    df.drop('Outlet_Identifier', axis=1, inplace=True)

    return df