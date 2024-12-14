from scipy.stats import alpha, f_oneway
from random import sample
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

if not os.path.exists(f'{dir_path}/Figures'):
    os.makedirs(f'{dir_path}/Figures')

train_df = pd.read_csv("./Datasets/train.csv", low_memory=False)
store_df = pd.read_csv("./Datasets/store.csv", low_memory=False)

merged_train_df = pd.merge(train_df, store_df, on="Store")
competitionDate_df = merged_train_df[
    ["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"]
].rename(
    columns={"CompetitionOpenSinceYear": "YEAR", "CompetitionOpenSinceMonth": "MONTH"}
)
competitionDate_df = pd.to_datetime(competitionDate_df[["YEAR", "MONTH"]].assign(DAY=1))
merged_train_df["CompetitionOpenSinceYear"] = competitionDate_df
merged_train_df.drop("CompetitionOpenSinceMonth", axis=1, inplace=True)
merged_train_df.rename(
    columns={"CompetitionOpenSinceYear": "CompetitionOpenSinceDate"}, inplace=True
)


temp_df = pd.DataFrame(index=merged_train_df.index)
temp_df["Promo2SinceYear"] = merged_train_df["Promo2SinceYear"].fillna(
    1900
)  # or choose another default year
temp_df["Promo2SinceWeek"] = merged_train_df["Promo2SinceWeek"].fillna(1).astype(int)
promo2sinceDate_df = pd.to_datetime(
    temp_df["Promo2SinceYear"].astype(str) + "101", format="%Y.%m%d"
)
promo2sinceDate_df += pd.to_timedelta(temp_df["Promo2SinceWeek"] - 1, unit="W")
merged_train_df["Promo2SinceYear"] = promo2sinceDate_df
merged_train_df.drop("Promo2SinceWeek", axis=1, inplace=True)
merged_train_df.rename(columns={"Promo2SinceYear": "Promo2SinceDate"}, inplace=True)
merged_train_df.loc[
    merged_train_df["Promo2SinceDate"].dt.year == 1900, "Promo2SinceDate"
] = None

merged_train_df["Promo2SinceDate"] = pd.to_datetime(merged_train_df["Promo2SinceDate"])
merged_train_df["Date"] = pd.to_datetime(merged_train_df["Date"])
merged_train_df["Promo2SinceDate"] = (
    merged_train_df["Date"] - merged_train_df["Promo2SinceDate"]
)
merged_train_df.rename(columns={"Promo2SinceDate": "Promo2ForDays"}, inplace=True)
merged_train_df["CompetitionOpenSinceDate"] = (
    merged_train_df["Date"] - merged_train_df["CompetitionOpenSinceDate"]
)
merged_train_df.rename(
    columns={"CompetitionOpenSinceDate": "CompetitionOpenForDays"}, inplace=True
)
merged_train_df["Promo2ForDays"] = merged_train_df[
    "Promo2ForDays"
].dt.days  # take n_days as integer NOT datetime
merged_train_df["CompetitionOpenForDays"] = merged_train_df[
    "CompetitionOpenForDays"
].dt.days  # take n_days as integer NOT datetime

indices_negative_promo = merged_train_df.query("Promo2ForDays < 0").index
indices_negative_competition = merged_train_df.query("CompetitionOpenForDays < 0").index
indices_sample_negative_promo = sample(list(indices_negative_promo), 5)
indices_sample_negative_competition = sample(list(indices_negative_competition), 5)

merged_train_df.drop("CompetitionOpenForDays", axis=1, inplace=True)
merged_train_df.dropna(subset=["CompetitionDistance"], inplace=True)

# First handling the negative values
merged_train_df.loc[merged_train_df["Promo2ForDays"] <= 0, "Promo2"] = 0
merged_train_df.loc[merged_train_df["Promo2"] == 0, "PromoInterval"] = "0"
merged_train_df.loc[merged_train_df["Promo2ForDays"] < 0, "Promo2ForDays"] = 0
# Then handling nan values
merged_train_df["Promo2ForDays"] = merged_train_df["Promo2ForDays"].fillna(0)
merged_train_df["PromoInterval"] = merged_train_df[["PromoInterval"]].fillna("0")


indices_zero_promo2 = merged_train_df.query('Promo2 == "0"').index
indices_zero_promo2fordays = merged_train_df.query("Promo2ForDays == 0").index
indices_zero_promointerval = merged_train_df.query('PromoInterval == "0"').index

merged_train_df.drop("Promo2", axis=1, inplace=True)

merged_train_df.rename(columns={"Date": "MonthOfYear"}, inplace=True)
merged_train_df["MonthOfYear"] = merged_train_df["MonthOfYear"].dt.month

merged_train_df.drop("Customers", axis=1, inplace=True)

merged_train_df = merged_train_df.astype(
    {"MonthOfYear": "str", "Open": "str", "Promo": "str"}
)

merged_train_df.hist(
    column="Sales", bins=100, grid=False, color="#86bf91", zorder=2, rwidth=0.9
)

plt.savefig(f'{dir_path}/Figures/Figure_1.png', bbox_inches='tight')

indices_closed_0sales = merged_train_df.query('Open == "0" and Sales == 0').index
indices_0sales = merged_train_df.query("Sales == 0").index
indices_closed = merged_train_df.query('Open == "0"').index

merged_train_df.drop(indices_closed_0sales, inplace=True)
merged_train_df.drop("Open", axis=1, inplace=True)

merged_train_df.hist(
    column="Sales", bins=100, grid=False, color="#86bf91", zorder=2, rwidth=0.9
)

plt.savefig(f'{dir_path}/Figures/Figure_2.png', bbox_inches='tight')

merged_train_df = merged_train_df.astype(
    {
        "Store": "str",
        "DayOfWeek": "str",
        "Sales": "float",
        "Promo": "str",
        "StateHoliday": "str",
        "SchoolHoliday": "str",
        "PromoInterval": "str",
    }
)


sns.set_style("darkgrid")
sns.set_palette("Set2")

figure_1 = plt.figure(figsize=(14, 9))
gs = figure_1.add_gridspec(3, 2)

ax_dayofweek = figure_1.add_subplot(gs[0, 0])
sns.boxplot(
    y="DayOfWeek",
    x="Sales",
    data=merged_train_df,
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
    data=merged_train_df,
    ax=ax_storetype,
    whis=(0, 100),
    width=0.5,
)
ax_storetype.set_title("Sales vs. StoreType", fontweight="bold")

ax_assortment = figure_1.add_subplot(gs[2, 0])
sns.boxplot(
    y="Assortment",
    x="Sales",
    data=merged_train_df,
    ax=ax_assortment,
    whis=(0, 100),
    width=0.5,
)
ax_assortment.set_title("Sales vs. Assortment", fontweight="bold")

ax_promo = figure_1.add_subplot(gs[2, 1])
sns.boxplot(
    y="PromoInterval",
    x="Sales",
    data=merged_train_df,
    ax=ax_promo,
    whis=(0, 100),
    width=0.5,
)
ax_promo.set_title("Sales vs. PromoInterval", fontweight="bold")

ax_monthofyear = figure_1.add_subplot(gs[:2, 1])
sns.boxplot(
    y="MonthOfYear",
    x="Sales",
    data=merged_train_df,
    order=[str(i + 1) for i in range(12)],
    ax=ax_monthofyear,
    whis=(0, 100),
    width=0.5,
)
ax_monthofyear.set_title("Sales vs. MonthOfYear", fontweight="bold")

plt.tight_layout()

plt.savefig(f'{dir_path}/Figures/Figure_3.png', bbox_inches='tight')

merged_train_df.drop("PromoInterval", axis=1, inplace=True)


figure_2, axis_2 = plt.subplots(nrows=3, ncols=1, figsize=(12, 7))
axis_2[0].hist(
    merged_train_df[merged_train_df["Promo"] == "1"]["Sales"],
    bins=100,
    alpha=0.3,
    color="green",
    label="Promo == 1",
)
axis_2[0].hist(
    merged_train_df[merged_train_df["Promo"] == "0"]["Sales"],
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
    merged_train_df[merged_train_df["SchoolHoliday"] == "1"]["Sales"],
    bins=100,
    alpha=0.3,
    color="green",
    label="SchoolHoliday == 1",
)
axis_2[1].hist(
    merged_train_df[merged_train_df["SchoolHoliday"] == "0"]["Sales"],
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
    merged_train_df[merged_train_df["StateHoliday"] == "1"]["Sales"],
    bins=100,
    alpha=0.3,
    color="green",
    label="StateHoliday == 1",
)
axis_2[2].hist(
    merged_train_df[merged_train_df["StateHoliday"] == "0"]["Sales"],
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

plt.savefig(f'{dir_path}/Figures/Figure_4.png', bbox_inches='tight')

merged_train_df.drop(["StateHoliday", "SchoolHoliday"], axis=1, inplace=True)

figure_3, axis_3 = plt.subplots(nrows=3, ncols=1, figsize=(12, 7))
axis_3[0].scatter(merged_train_df["Promo2ForDays"], merged_train_df["Sales"])
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
    merged_train_df[merged_train_df["Assortment"] == "a"]["CompetitionDistance"],
    merged_train_df[merged_train_df["Assortment"] == "a"]["Sales"],
    c="blue",
    alpha=0.2,
    s=15,
    marker="o",
)

axis_3[1].scatter(
    merged_train_df[merged_train_df["Assortment"] == "b"]["CompetitionDistance"],
    merged_train_df[merged_train_df["Assortment"] == "b"]["Sales"],
    c="magenta",
    alpha=0.2,
    s=15,
    marker="o",
)

axis_3[1].scatter(
    merged_train_df[merged_train_df["Assortment"] == "c"]["CompetitionDistance"],
    merged_train_df[merged_train_df["Assortment"] == "c"]["Sales"],
    c="cyan",
    alpha=0.2,
    s=15,
    marker="o",
)

axis_3[1].set_xlabel("CompetitionDistance")
axis_3[1].set_ylabel("Sales")

color_patches = [
    matplotlib.lines.Line2D(
        [0],
        [0],
        marker=".",
        color="w",
        label="Assortment: a",
        markerfacecolor="blue",
        markersize=15,
    ),
    matplotlib.lines.Line2D(
        [0],
        [0],
        marker=".",
        color="w",
        label="Assortment: b",
        markerfacecolor="magenta",
        markersize=15,
    ),
    matplotlib.lines.Line2D(
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
    merged_train_df[merged_train_df["StoreType"] == "a"]["CompetitionDistance"],
    merged_train_df[merged_train_df["StoreType"] == "a"]["Sales"],
    c="blue",
    alpha=0.2,
    s=15,
    marker="o",
)

axis_3[2].scatter(
    merged_train_df[merged_train_df["StoreType"] == "b"]["CompetitionDistance"],
    merged_train_df[merged_train_df["StoreType"] == "b"]["Sales"],
    c="green",
    alpha=0.2,
    s=15,
    marker="o",
)

axis_3[2].scatter(
    merged_train_df[merged_train_df["StoreType"] == "c"]["CompetitionDistance"],
    merged_train_df[merged_train_df["StoreType"] == "c"]["Sales"],
    c="cyan",
    alpha=0.2,
    s=15,
    marker="o",
)

axis_3[2].scatter(
    merged_train_df[merged_train_df["StoreType"] == "d"]["CompetitionDistance"],
    merged_train_df[merged_train_df["StoreType"] == "d"]["Sales"],
    c="magenta",
    alpha=0.2,
    s=15,
    marker="o",
)

axis_3[2].set_xlabel("CompetitionDistance")
axis_3[2].set_ylabel("Sales")

color_patches = [
    matplotlib.lines.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="StoreType: a",
        markerfacecolor="blue",
        markersize=10,
    ),
    matplotlib.lines.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="StoreType: b",
        markerfacecolor="green",
        markersize=10,
    ),
    matplotlib.lines.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="StoreType: c",
        markerfacecolor="cyan",
        markersize=10,
    ),
    matplotlib.lines.Line2D(
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

plt.savefig(f'{dir_path}/Figures/Figure_5.png', bbox_inches='tight')

numeric_df = merged_train_df[["Sales", "CompetitionDistance", "Promo2ForDays"]]
numeric_correlations = numeric_df.corr()
plt.figure(figsize=(9, 7))
sns.heatmap(numeric_correlations, cmap="RdBu_r", annot=True)
plt.title("Correlation of Sales, Competition Distance, and Promo2 Days", fontsize=16)
plt.savefig('Figure_6.png', bbox_inches='tight')

merged_train_df.drop("Promo2ForDays", axis=1, inplace=True)
store_sales_list = [
    list(merged_train_df.loc[merged_train_df["Store"] == label, "Sales"])
    for label in set(merged_train_df["Store"])
]
F, p = f_oneway(*store_sales_list)
print(F, p)
