import pandas as pd
from fontTools.ttx import process
from scipy.stats import alpha, f_oneway
from sqlalchemy import create_engine
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import json


def visual_eda(processed_train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs visual exploratory data analysis (EDA) using bin-, scatter- and box- plots
    between features and the target variable (e.g., Sales). The information from plots is used to
    drop certain data based in information gained from plots.

    Parameters:
    -----------
    processed_train_df : pd.DataFrame
        The input DataFrame containing the processed training data, including features and
        the target variable.

    Returns:
    --------
    pd.DataFrame
        A slimmed down DataFrame without unnecessary features or samples.
    """

    """
    We explore the relation between the features and the target feature (Sales) visually, this step is performed
    before conducting any statistical analysis as it is easier to conduct an assessment visually than statistically
    since the latter may require certain assumptions like having a dataset of Gaussian distribution.
      """

    """ we take a look at the distribution of values (Sales) """
    processed_train_df.hist(
        column="Sales", bins=100, grid=False, color="#86bf91", zorder=2, rwidth=0.9
    )
    # plt.show()

    """we notice that there are many small values, maybe they are just 0 sales for when the store is closed!"""
    indices_closed_0sales = processed_train_df.query('Open == "0" and Sales == 0').index
    indices_0sales = processed_train_df.query("Sales == 0").index
    indices_closed = processed_train_df.query('Open == "0"').index
    print("n_samples Days closed AND Sales=0: ", len(list(indices_closed_0sales)))
    print("n_samples Sales=0: ", len(list(indices_0sales)))
    print("n_samples Open=0: ", len(list(indices_closed)))
    print(
        "Identical"
        if list(indices_closed) == list(indices_closed_0sales)
        else "Not Identical"
    )

    """since indices_closed == indices_closed_0sales, we see that nearly all the samples with 0 sales
    are ones where the stores are simply closed with Open=0
       Therefore, we remove the rows with Open=0 and remove the column (open) as it is not interesting"""
    processed_train_df.drop(indices_closed_0sales, inplace=True)
    processed_train_df.drop("Open", axis=1, inplace=True)

    """ with the histogram we can see that:
        1- The peak at 0 is gone. 
        2- The distribution of Sales data is !close! to being normal / Gaussian which
        may allow the use ANOVA (which requires normality) without worrying about violating this assumption"""
    processed_train_df.hist(
        column="Sales", bins=100, grid=False, color="#86bf91", zorder=2, rwidth=0.9
    )
    # plt.show()

    """Next, we explore the relation between our target variable (Sales) and some of the categorical features
       namely, (DayOfWeek, MonthOfYear, Assortment, StoreType)"""
    sns.set_style("darkgrid")
    sns.set_palette("Set2")

    figure_1 = plt.figure(figsize=(14, 9))
    gs = figure_1.add_gridspec(3, 2)

    ax_dayofweek = figure_1.add_subplot(gs[0, 0])
    sns.boxplot(
        y="DayOfWeek",
        x="Sales",
        data=processed_train_df,
        order=[str(i + 1) for i in range(7)],
        ax=ax_dayofweek,
        whis=(0, 100),
        width=0.5,
    )
    ax_dayofweek.set_title("Sales vs. DayOfWeek", fontweight="bold")

    ax_storetype = figure_1.add_subplot(gs[1, 0])
    sns.boxplot(
        y="StoreType",
        x="Sales",
        data=processed_train_df,
        ax=ax_storetype,
        whis=(0, 100),
        width=0.5,
    )
    ax_storetype.set_title("Sales vs. StoreType", fontweight="bold")

    ax_assortment = figure_1.add_subplot(gs[2, 0])
    sns.boxplot(
        y="Assortment",
        x="Sales",
        data=processed_train_df,
        ax=ax_assortment,
        whis=(0, 100),
        width=0.5,
    )
    ax_assortment.set_title("Sales vs. Assortment", fontweight="bold")

    ax_promo = figure_1.add_subplot(gs[2, 1])
    sns.boxplot(
        y="PromoInterval",
        x="Sales",
        data=processed_train_df,
        ax=ax_promo,
        whis=(0, 100),
        width=0.5,
    )
    ax_promo.set_title("Sales vs. PromoInterval", fontweight="bold")

    ax_monthofyear = figure_1.add_subplot(gs[:2, 1])
    sns.boxplot(
        y="MonthOfYear",
        x="Sales",
        data=processed_train_df,
        order=[str(i + 1) for i in range(12)],
        ax=ax_monthofyear,
        whis=(0, 100),
        width=0.5,
    )
    ax_monthofyear.set_title("Sales vs. MonthOfYear", fontweight="bold")

    plt.tight_layout()
    # plt.show()
    """ We can see that in the features (DayOfWeek, MonthOfYear, Assortment, StoreType) at least one label 
        in each of these features deviates in its range of values in Sales which means that these features 
        will all be useful. However, PromoInterval offers no predictive power and so it is removed."""
    processed_train_df.drop("PromoInterval", axis=1, inplace=True)

    """ Next, we look into the relation between Sales and the features: (promo, SchoolHoliday, StateHoliday)"""
    figure_2, axis_2 = plt.subplots(nrows=3, ncols=1, figsize=(12, 7))
    axis_2[0].hist(
        processed_train_df[processed_train_df["Promo"] == "1"]["Sales"],
        bins=100,
        alpha=0.3,
        color="green",
        label="Promo == 1",
    )
    axis_2[0].hist(
        processed_train_df[processed_train_df["Promo"] == "0"]["Sales"],
        bins=100,
        alpha=0.3,
        color="blue",
        label="Promo == 0",
    )
    axis_2[0].set_xlabel("Sales")
    axis_2[0].set_ylabel("n_Samples")
    axis_2[0].text(
        0.5,
        0.9,
        "Sales vs. Promo",
        ha="center",
        va="center",
        transform=axis_2[0].transAxes,
        fontsize=11,
        fontweight="bold",
    )
    axis_2[0].legend()

    axis_2[1].hist(
        processed_train_df[processed_train_df["SchoolHoliday"] == "1"]["Sales"],
        bins=100,
        alpha=0.3,
        color="green",
        label="SchoolHoliday == 1",
    )
    axis_2[1].hist(
        processed_train_df[processed_train_df["SchoolHoliday"] == "0"]["Sales"],
        bins=100,
        alpha=0.3,
        color="blue",
        label="SchoolHoliday == 0",
    )
    axis_2[1].set_xlabel("Sales")
    axis_2[1].set_ylabel("n_Samples")
    axis_2[1].text(
        0.5,
        0.9,
        "Sales vs. SchoolHoliday",
        ha="center",
        va="center",
        transform=axis_2[1].transAxes,
        fontsize=11,
        fontweight="bold",
    )
    axis_2[1].legend()

    axis_2[2].hist(
        processed_train_df[processed_train_df["StateHoliday"] == "1"]["Sales"],
        bins=100,
        alpha=0.3,
        color="green",
        label="StateHoliday == 1",
    )
    axis_2[2].hist(
        processed_train_df[processed_train_df["StateHoliday"] == "0"]["Sales"],
        bins=100,
        alpha=0.3,
        color="blue",
        label="StateHoliday == 0",
    )
    axis_2[2].set_xlabel("Sales")
    axis_2[2].set_ylabel("n_Samples")
    axis_2[2].text(
        0.5,
        0.9,
        "Sales vs. StateHoliday",
        ha="center",
        va="center",
        transform=axis_2[2].transAxes,
        fontsize=11,
        fontweight="bold",
    )
    axis_2[2].legend()

    """
    Observations:
    plot-1- Knowing if there is a promotion clearly helps in making a prediction given the difference in distributions
    plot_2- There seems to be no predictive power in SchoolHoliday as the distribution is almost the same 
    whether there is a SchoolHoliday or not apart from the ability to indicate the potential presence of outliers.
    plot_3- We can see that having a StateHoliday only has samples with StateHoliday == 0, this could be that the store always
        closes when there is a StateHoliday (StateHoliday==1), and because we have removed all the samples with 
        Open==0, then what we have left are the samples only with StateHoliday==0

    Decisions:
    1- Remove the columns SchoolHoliday and StateHoliday
     """
    processed_train_df.drop(["StateHoliday", "SchoolHoliday"], axis=1, inplace=True)
    # explore_df(processed_train_df)

    """Next, we look into scatter plots between Sales and (Promo2ForDays, CompetitionDistance)"""
    figure_3, axis_3 = plt.subplots(nrows=3, ncols=1, figsize=(12, 7))
    axis_3[0].scatter(processed_train_df["Promo2ForDays"], processed_train_df["Sales"])
    axis_3[0].set_xlabel("Promo2ForDays")
    axis_3[0].set_ylabel("Sales")
    axis_3[0].text(
        0.5,
        0.9,
        "Sales vs. Promo2ForDays",
        ha="center",
        va="center",
        transform=axis_3[0].transAxes,
        fontsize=11,
        fontweight="bold",
    )

    axis_3[1].scatter(
        processed_train_df[processed_train_df["Assortment"] == "a"][
            "CompetitionDistance"
        ],
        processed_train_df[processed_train_df["Assortment"] == "a"]["Sales"],
        c="blue",
        alpha=0.2,
        s=15,
        marker="o",
    )

    axis_3[1].scatter(
        processed_train_df[processed_train_df["Assortment"] == "b"][
            "CompetitionDistance"
        ],
        processed_train_df[processed_train_df["Assortment"] == "b"]["Sales"],
        c="magenta",
        alpha=0.2,
        s=15,
        marker="o",
    )

    axis_3[1].scatter(
        processed_train_df[processed_train_df["Assortment"] == "c"][
            "CompetitionDistance"
        ],
        processed_train_df[processed_train_df["Assortment"] == "c"]["Sales"],
        c="cyan",
        alpha=0.2,
        s=15,
        marker="o",
    )

    axis_3[1].set_xlabel("CompetitionDistance")
    axis_3[1].set_ylabel("Sales")

    color_patches = [
        Line2D(
            [0],
            [0],
            marker=".",
            color="w",
            label="Assortment: a",
            markerfacecolor="blue",
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            marker=".",
            color="w",
            label="Assortment: b",
            markerfacecolor="magenta",
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            marker=".",
            color="w",
            label="Assortment: c",
            markerfacecolor="cyan",
            markersize=15,
        ),
    ]

    axis_3[1].legend(handles=color_patches)

    axis_3[1].text(
        0.5,
        0.9,
        "Sales vs. CompetitionDistance, with Assortment",
        ha="center",
        va="center",
        transform=axis_3[1].transAxes,
        fontsize=11,
        fontweight="bold",
    )

    axis_3[2].scatter(
        processed_train_df[processed_train_df["StoreType"] == "a"][
            "CompetitionDistance"
        ],
        processed_train_df[processed_train_df["StoreType"] == "a"]["Sales"],
        c="blue",
        alpha=0.2,
        s=15,
        marker="o",
    )

    axis_3[2].scatter(
        processed_train_df[processed_train_df["StoreType"] == "b"][
            "CompetitionDistance"
        ],
        processed_train_df[processed_train_df["StoreType"] == "b"]["Sales"],
        c="green",
        alpha=0.2,
        s=15,
        marker="o",
    )

    axis_3[2].scatter(
        processed_train_df[processed_train_df["StoreType"] == "c"][
            "CompetitionDistance"
        ],
        processed_train_df[processed_train_df["StoreType"] == "c"]["Sales"],
        c="cyan",
        alpha=0.2,
        s=15,
        marker="o",
    )

    axis_3[2].scatter(
        processed_train_df[processed_train_df["StoreType"] == "d"][
            "CompetitionDistance"
        ],
        processed_train_df[processed_train_df["StoreType"] == "d"]["Sales"],
        c="magenta",
        alpha=0.2,
        s=15,
        marker="o",
    )

    axis_3[2].set_xlabel("CompetitionDistance")
    axis_3[2].set_ylabel("Sales")

    color_patches = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="StoreType: a",
            markerfacecolor="blue",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="StoreType: b",
            markerfacecolor="green",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="StoreType: c",
            markerfacecolor="cyan",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="StoreType: d",
            markerfacecolor="magenta",
            markersize=10,
        ),
    ]

    axis_3[2].legend(handles=color_patches)
    axis_3[2].text(
        0.5,
        0.9,
        "Sales vs. CompetitionDistance, with StoreType",
        ha="center",
        va="center",
        transform=axis_3[2].transAxes,
        fontsize=11,
        fontweight="bold",
    )
    """Observations:
    plot_1- There seems to be no correlation between Promo2ForDays and Sales
    plot_2- and plot_3- There seems to be a negative correlation as the greater CompetitionDistance is, the lower the Sales

    Decisions:
    1- We consider Dropping the column Promo2ForDays, but wait to see the heatmap of correlations later on...

    Side note:
    It is interesting to see how certain Assortment and/or StoreTypes exist depending on
    CompetitionDistance"""
    # plt.show()
    return processed_train_df


def statistical_eda(processed_train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Statistical Analysis of relation between features and target. Performs in order:

    1. Plots the correlation matrix to identify relationships between features.
    2. Performs ANOVA (Analysis of Variance) for the feature with high cardinality to
       determine its impact on the target variable.
    3. Applies compression to high-cardinality features based on the findings, reducing
       dimensionality while retaining significant information.

    Parameters:
    -----------
    data : pd.DataFrame
        The input DataFrame containing the features to be analyzed and compressed.

    Returns:
    --------
    data : pd.DataFrame
    """

    """ Part A:  look into correlation between Sales and  (Promo2ForDays, CompetitionDistance) """
    numeric_df = processed_train_df[["Sales", "CompetitionDistance", "Promo2ForDays"]]
    numeric_correlations = numeric_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_correlations, cmap="RdBu_r", annot=True)

    """Observations:
    - The correlation is weak between Sales and (CompetitionDistance, Promo2ForDays),  
       Decisions:
    - We remove only Promo2ForDays, when it comes to CompetitionDistance, we keep it since as we saw earlier in
    visual_eda CompetitionDistance relates to Assortment and StorType which correlate with Sales, so the correlation
    between CompetitionDistance and Sales is indirect and is not visible using a simple correlation measure.
    """

    processed_train_df.drop("Promo2ForDays", axis=1, inplace=True)

    """Now we implement ANOVA which is the alternative to a the boxplot we had earlier for features with 
    many labels like Store with about 1115 labels. ANOVA tells us if the feature (Store) has any predictive value 
    for Sales"""

    store_sales_list = [
        list(processed_train_df.loc[processed_train_df["Store"] == label, "Sales"])
        for label in set(processed_train_df["Store"])
    ]
    F, p = f_oneway(*store_sales_list)
    print(F, p)
    """ the result is F = 1144 and p=0.0 which means that there is at least one label whose Sales distribution 
        deviates from the other labels' distributions which means that we can keep Store column since it has good
        predictive power."""

    """Because Store has 1115 labels, we have to perform some compression. Something as simple one-hot encoding would
     make the samples very sparse in the feature space causing difficulties in learning and prediction.
      
       Embedding could be used but is not reasonable in this application as it is usually used in the context of NLP 
    and neither is regularization as it is used to tune down features of less importance.
    
        We choose to handle this intuitively, the Store label is used to give an idea over possible Sales range, 
    therefore, we divide the Sales dimension into n bins, where each bin is a new label of Store, then for each
     Store (1, 2, 3...) we calculate the median of its sales and check under which bin it falls, then 
     give it a new label following bin (say Store number 27 --> bin number 1). This guides the possible Sales value"""

    # Median Sales of each label
    median_sales_per_label = processed_train_df.groupby("Store")["Sales"].median()
    median_sales_per_label.hist(
        bins=15, grid=False, color="#86bf91", zorder=2, rwidth=0.9
    )
    # divide Store into bins
    binned_medians = pd.cut(median_sales_per_label, bins=20, labels=range(20))  # was 20
    binned_dict = binned_medians.to_dict()

    # Map original Store number into corresponding bin
    processed_train_df[f"{'Store'}_compressed"] = processed_train_df["Store"].map(
        binned_medians
    )
    processed_train_df.drop("Store", axis=1, inplace=True)
    # plt.show()
    return processed_train_df
