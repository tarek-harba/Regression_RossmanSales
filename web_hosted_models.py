import streamlit as st
import json
import torch
import joblib
import numpy as np
import pandas as pd
import regression_forest
from sklearn.preprocessing import OneHotEncoder

st.title("Predict your own Store Sales!")
st.write(
    "Using the dials, toggles and input boxes below, you can make a prediction of a store's sales."
)
st.write(
    "Note that there are no units for any of the values because the units were not mentioned in the source of the data."
)


model_choice = st.radio(
    "Model used for prediction:", ["Deep Neural Network", "Regression Forest"]
)
assortment = st.selectbox(
    "Assortment: which set of goods the store offers.", ["a", "b", "c"]
)
storetype = st.selectbox(
    "StoreType: Varies by location and size.", ["a", "b", "c", "d"]
)
promo = st.toggle("Ongoing Promotion")

day = st.selectbox("Day of Week:", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
day_dict = {
    "Mon": 1,
    "Tue": 2,
    "Wed": 3,
    "Thu": 4,
    "Fri": 5,
    "Sat": 6,
    "Sun": 7,
}

day_st = str(day_dict[day])
month = st.number_input("Month of Year:", 1, 12)
dist = st.slider("Distance to nearest competitor:", 0, 75860)
dist = (dist - 5459.664289865578) / 7815.189957479083
store = st.select_slider(
    "Store Number: specify store in chain.", [str(i + 1) for i in range(1115)]
)
with open("./Misc_Files/Store_dict.json", "r") as fp:
    store_dict = json.load(fp)
store_nr = store_dict[store]


if model_choice == "Deep Neural Network":
    og_sample = {
        "DayOfWeek": str(day_st),
        "MonthOfYear": str(month),
        "Promo": str(int(promo)),
        "StoreType": str(storetype),
        "Assortment": str(assortment),
        "CompetitionDistance": dist,
        "Store_compressed": str(store_nr),
    }
    sample = []
    for col_name, value in og_sample.items():
        if col_name != "CompetitionDistance":
            with open(f"./Misc_Files/{col_name}.json", "r") as fp:
                col_dict = json.load(fp)
            col_value = int(col_dict[value])
            ohe = [0] * len(col_dict)
            ohe[col_value] = 1
            sample.extend(ohe)
        else:
            sample.extend([value])

    sample = torch.Tensor(sample)
    model = torch.load(r"./Trained_Models/DNN.pth", weights_only=False)
    model.eval()
    prediction = model(sample)
    st.success(f"The store Sales should be around: {str(int(prediction.item()))}")
else:
    ref_dict = {
        "CompetitionDistance": 0,
        "Assortment_1": 0.0,
        "Assortment_2": 0.0,
        "Assortment_3": 0.0,
        "StoreType_1": 0.0,
        "StoreType_2": 0.0,
        "StoreType_3": 0.0,
        "StoreType_4": 0.0,
        "Promo_0": 0.0,
        "Promo_1": 0.0,
        "MonthOfYear_1": 0.0,
        "MonthOfYear_10": 0.0,
        "MonthOfYear_11": 0.0,
        "MonthOfYear_12": 0.0,
        "MonthOfYear_2": 0.0,
        "MonthOfYear_3": 0.0,
        "MonthOfYear_4": 0.0,
        "MonthOfYear_5": 0.0,
        "MonthOfYear_6": 0.0,
        "MonthOfYear_7": 0.0,
        "MonthOfYear_8": 0.0,
        "MonthOfYear_9": 0.0,
        "DayOfWeek_1": 0.0,
        "DayOfWeek_2": 0.0,
        "DayOfWeek_3": 0.0,
        "DayOfWeek_4": 0.0,
        "DayOfWeek_5": 0.0,
        "DayOfWeek_6": 0.0,
        "DayOfWeek_7": 0.0,
        "Store_compressed_0": 0.0,
        "Store_compressed_1": 0.0,
        "Store_compressed_2": 0.0,
        "Store_compressed_3": 0.0,
        "Store_compressed_4": 0.0,
        "Store_compressed_5": 0.0,
        "Store_compressed_6": 0.0,
        "Store_compressed_7": 0.0,
        "Store_compressed_8": 0.0,
        "Store_compressed_9": 0.0,
        "Store_compressed_10": 0.0,
        "Store_compressed_11": 0.0,
        "Store_compressed_12": 0.0,
        "Store_compressed_13": 0.0,
        "Store_compressed_14": 0.0,
        "Store_compressed_15": 0.0,
        "Store_compressed_16": 0.0,
        "Store_compressed_17": 0.0,
        "Store_compressed_18": 0.0,
        "Store_compressed_19": 0.0,
    }

    sample_dict = {
        "DayOfWeek": str(day_st),
        "MonthOfYear": str(month),
        "Promo": str(int(promo)),
        "StoreType": str(storetype),
        "Assortment": str(assortment),
        "CompetitionDistance": dist,
        "Store_compressed": str(store_nr),
    }

    for key, value in sample_dict.items():
        if key != "CompetitionDistance":
            with open(f"./Misc_Files/{key}.json", "r") as fp:
                col_dict = json.load(fp)
            col_value = int(col_dict[value])
            if (
                key == "StoreType" or key == "Assortment" or key == "DayOfWeek"
            ):  # str starts with 1
                key_string = f"{str(key)}_{str(col_value + 1)}"
                ref_dict[key_string] = 1
            if key == "Store_compressed" or key == "Promo":  # str starts with 0
                key_string = f"{str(key)}_{str(col_value)}"
                ref_dict[key_string] = 1
            if key == "MonthOfYear":
                key_string = f"{str(key)}_{str(value)}"
                ref_dict[key_string] = 1
        else:
            ref_dict["CompetitionDistance"] = value

    loaded_rf = joblib.load(r"./Trained_Models/regression_forest.joblib")
    prediction = loaded_rf.predict(pd.DataFrame([ref_dict]))
    st.success(f"The store Sales should be around: {str(int(prediction))}")
