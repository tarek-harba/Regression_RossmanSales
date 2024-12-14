import pandas as pd


def data_cleanup(complete_train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses the given Pandas DataFrame by performing the following operations:

    1. Drops unnecessary columns that are not relevant for analysis or modeling.
    2. Reformats data to simplify later work.
    3. Handles missing values.

    Parameters:
    -----------
    complete_train_df : pd.DataFrame
        The input DataFrame that contains the raw data to be cleaned and preprocessed.

    Returns:
    --------
    pd.DataFrame
        A cleaned and preprocessed DataFrame ready for further analysis.
    """

    """ the columns (CompetitionOpenSinceMonth,CompetitionOpenSinceYear) can be reduced
     to a single column of (CompetitionOpenSinceDate). Reducing n_dimensions, note that we DO have some None values"""

    competitionDate_df = complete_train_df[
        ["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"]
    ].rename(
        columns={
            "CompetitionOpenSinceYear": "YEAR",
            "CompetitionOpenSinceMonth": "MONTH",
        }
    )
    competitionDate_df = pd.to_datetime(
        competitionDate_df[["YEAR", "MONTH"]].assign(DAY=1)
    )
    complete_train_df["CompetitionOpenSinceYear"] = competitionDate_df
    complete_train_df.drop("CompetitionOpenSinceMonth", axis=1, inplace=True)
    complete_train_df.rename(
        columns={"CompetitionOpenSinceYear": "CompetitionOpenSinceDate"}, inplace=True
    )
    # print(complete_train_df.head(10))
    # print(u'\u2500' * 200)

    """ the columns (Promo2SinceWeek, Promo2SinceYear) can be reduced
    #  to a single column of (Promo2SinceDate). Reducing n_dimensions, note that we DO have some None values"""

    temp_df = pd.DataFrame(index=complete_train_df.index)
    temp_df["Promo2SinceYear"] = complete_train_df["Promo2SinceYear"].fillna(
        1900
    )  # or choose another default year
    temp_df["Promo2SinceWeek"] = (
        complete_train_df["Promo2SinceWeek"].fillna(1).astype(int)
    )
    promo2sinceDate_df = pd.to_datetime(
        temp_df["Promo2SinceYear"].astype(str) + "0101", format="%Y%m%d"
    )
    promo2sinceDate_df += pd.to_timedelta(temp_df["Promo2SinceWeek"] - 1, unit="W")

    complete_train_df["Promo2SinceYear"] = promo2sinceDate_df
    complete_train_df.drop("Promo2SinceWeek", axis=1, inplace=True)
    complete_train_df.rename(
        columns={"Promo2SinceYear": "Promo2SinceDate"}, inplace=True
    )
    complete_train_df.loc[
        complete_train_df["Promo2SinceDate"].dt.year == 1900, "Promo2SinceDate"
    ] = None
    # print(complete_train_df.head(20))
    # print(u'\u2500' * 200)

    """ The Columns CompetitionOpenSinceDate and Promo2SinceDate can be made more useful by converting them to 
    number of days from the current date in column Date under new names of (CompetitionOpenForDays / Promo2ForDays)"""

    complete_train_df["Promo2SinceDate"] = (
        complete_train_df["Date"] - complete_train_df["Promo2SinceDate"]
    )
    complete_train_df.rename(columns={"Promo2SinceDate": "Promo2ForDays"}, inplace=True)
    complete_train_df["CompetitionOpenSinceDate"] = (
        complete_train_df["Date"] - complete_train_df["CompetitionOpenSinceDate"]
    )
    complete_train_df.rename(
        columns={"CompetitionOpenSinceDate": "CompetitionOpenForDays"}, inplace=True
    )
    complete_train_df["Promo2ForDays"] = complete_train_df[
        "Promo2ForDays"
    ].dt.days  # take n_days as integer NOT datetime
    complete_train_df["CompetitionOpenForDays"] = complete_train_df[
        "CompetitionOpenForDays"
    ].dt.days  # take n_days as integer NOT datetime
    # print(complete_train_df.head(20))
    # print(u'\u2500' * 200)

    """ Just to ensure what we have makes sense, we look for negative Day values"""
    # indices_negative_promo = complete_train_df.query('Promo2ForDays < 0').index
    # indices_negative_competition = complete_train_df.query('CompetitionOpenForDays < 0').index
    # indices_sample_negative_promo = sample(list(indices_negative_promo), 5)
    # indices_sample_negative_competition = sample(list(indices_negative_competition), 5)
    # print(complete_train_df.loc[indices_sample_negative_promo])
    # print(u'\u2500' * 200)
    # print(complete_train_df.loc[indices_sample_negative_competition])
    # print(u'\u2500' * 200)

    """ We do have negative values in Promo2ForDays and CompetitionOpenForDays because their dates of occurrence are
        after the current Date. 
    We handle each column separately since dealing with each means that the neighboring columns,
    namely CompetitionDistance and Promo2 will also have to be dealt with"""

    """Starting with CompetitionOpenForDays and CompetitionDistance"""
    # print("Neg CompetitionOpenForDays",
    #       len(list(complete_train_df.query('CompetitionOpenForDays < 0').index))/complete_train_df.shape[0])
    # print("Null CompetitionOpenForDays",
    #       complete_train_df["CompetitionOpenForDays"].isnull().sum()/complete_train_df.shape[0])
    # print("Null CompetitionDistance",
    #       complete_train_df["CompetitionDistance"].isnull().sum() / complete_train_df.shape[0])

    """Almost 31 percent of rows have null CompetitionOpenForDays, there is no reasonable way to impute these
       as they do NOT correlate with any other variable and attempting to impute is irrational.
    Only 0.26 percent of the rows have null CompetitionDistance, these rows are dropped as they are too few
    and imputation is not done for the same reason as before.
    """
    complete_train_df.drop("CompetitionOpenForDays", axis=1, inplace=True)
    complete_train_df.dropna(subset=["CompetitionDistance"], inplace=True)

    """Moving on to Promo2 and Promo2ForDays"""
    # First handling the negative values
    complete_train_df.loc[complete_train_df["Promo2ForDays"] <= 0, "Promo2"] = 0
    complete_train_df.loc[complete_train_df["Promo2"] == 0, "PromoInterval"] = "0"
    complete_train_df.loc[complete_train_df["Promo2ForDays"] < 0, "Promo2ForDays"] = 0
    # Then handling nan values
    complete_train_df["Promo2ForDays"] = complete_train_df["Promo2ForDays"].fillna(0)
    complete_train_df["PromoInterval"] = complete_train_df[["PromoInterval"]].fillna(
        "0"
    )
    # print(complete_train_df.info())
    # print(complete_train_df.head(10))

    """ we notice that when promo2 is 0, Promo2ForDays is also 0, we ensure that
    if Promo2 is 0 then Promo2ForDays is 0, if so, we drop column Promo2 since it is redundant"""

    complete_train_df["Promo2"] = complete_train_df["Promo2"].astype(str)
    indices_zero_promo2 = complete_train_df.query('Promo2 == "0"').index
    indices_zero_promo2fordays = complete_train_df.query("Promo2ForDays == 0").index
    indices_zero_promointerval = complete_train_df.query('PromoInterval == "0"').index
    # print('Identical' if list(indices_zero_promo2fordays) == list(indices_zero_promo2) else "Not Identical")
    # print('Identical' if list(indices_zero_promo2fordays) == list(indices_zero_promointerval) else "Not Identical")

    """ We can see that the rows that have Promo2 == 0 also have PromoInterval AND Promo2Fordays == 0
    We can safely ignore promo2 column"""

    complete_train_df.drop("Promo2", axis=1, inplace=True)

    """ One more thing to adjust is the Date column, to make predictions later, the complete date is not actionable 
        (not integer and hard to categorize) and so we only take in month number which is suitable along 
        with the other column of 'DayOfWeek' where both help make predictions in certain days of week or 
        certain weeks in year (weeks in summer...)"""

    """ We could have used week number but then we would have near 52 categories which makes 
        the problem too sparse, whereas having just 12 months as categories is more approachable
        Plus we have other data like State/School Holiday which reduces importance of exact week number"""

    complete_train_df.rename(columns={"Date": "MonthOfYear"}, inplace=True)
    complete_train_df["MonthOfYear"] = complete_train_df["MonthOfYear"].dt.month
    # print(dir(complete_train_df["WeekOfYear"].dt))
    # print(complete_train_df.sample(10))

    """ Exploring the data set again"""
    # explore_df(complete_train_df)
    # print(complete_train_df.isnull().values.any())

    """We see that no missing values exist, the finishing touch is to remove
    the customers column as it is not known beforehand and we only seek to predict the Sales"""

    complete_train_df.drop("Customers", axis=1, inplace=True)

    """ Lastly we change the data that is categorical from int or float to string"""

    complete_train_df = complete_train_df.astype(
        {"MonthOfYear": "str", "Open": "str", "Promo": "str"}
    )
    # explore_df(complete_train_df)

    return complete_train_df
