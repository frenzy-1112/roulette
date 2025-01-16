#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests

# Load the saved Naive Bayes model
with open('naive_bayes_best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to fetch the last 10 spins from the API
def fetch_last_10_spins():
    base_url = "https://api.casinoscores.com/svc-evolution-game-events/api/autoroulette"
    params = {
        "size": 10,  # Number of results to fetch
        "sort": "data.settledAt,desc",
        "duration": 1
    }
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        numbers = [
            event["data"]["result"]["outcome"]["number"]
            for event in data
            if "data" in event and "result" in event["data"] and "outcome" in event["data"]["result"]
        ]
        return numbers
    else:
        st.error(f"Failed to fetch data. Status code: {response.status_code}")
        return []

# Function to extract features for each number
def get_roulette_features(number):
    if number is None:
        return "Green (0)", "Green", "-", "-"
    
    if 1 <= number <= 12:
        dozen = "Dozen 1 (1-12)"
    elif 13 <= number <= 24:
        dozen = "Dozen 2 (13-24)"
    elif 25 <= number <= 36:
        dozen = "Dozen 3 (25-36)"
    else:
        dozen = "Green (0)"
    
    red_numbers = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
    if number == 0:
        color = "Green"
    elif number in red_numbers:
        color = "Red"
    else:
        color = "Black"
    
    parity = "Even" if number % 2 == 0 else "Odd"
    high_low = "Low (1-18)" if 1 <= number <= 18 else "High (19-36)"
    
    return dozen, color, parity, high_low

# Streamlit UI
st.title("Roulette Prediction Model")

# Fetch and display the last 10 spins
st.header("Last 10 Spins with Features (Fetched from API)")
last_10_spins = fetch_last_10_spins()

if last_10_spins:
    # Reverse the order of the spins
    last_10_spins.reverse()

    # Extract features for the last 10 spins
    spin_data = []
    for num in last_10_spins:
        dozen, color, parity, high_low = get_roulette_features(num)
        spin_data.append({
            "Number": num,
            "Dozen": dozen,
            "Color": color,
            "Parity": parity,
            "High/Low": high_low
        })
    
    # Convert the data to a pandas DataFrame
    spins_df = pd.DataFrame(spin_data)
    
    # Display the DataFrame as a table
    st.table(spins_df)
    
    # Prepare input features for the model
    dozen_mapping = {"Dozen 1 (1-12)": 0, "Dozen 2 (13-24)": 1, "Dozen 3 (25-36)": 2, "Green (0)": 3}
    color_mapping = {"Red": 1, "Black": 0, "Green": 2}
    parity_mapping = {"Even": 1, "Odd": 0}
    high_low_mapping = {"Low (1-18)": 1, "High (19-36)": 0}

    dozen_lags = spins_df["Dozen"].apply(lambda x: dozen_mapping.get(x, 3))  # Default to 3 for 'Green (0)'
    color_lags = spins_df["Color"].apply(lambda x: color_mapping[x])
    parity_lags = spins_df["Parity"].apply(lambda x: parity_mapping[x])
    high_low_lags = spins_df["High/Low"].apply(lambda x: high_low_mapping[x])

    # Ensure that only 40 features are passed for prediction (remove the last_10_spins numbers)
    input_features = np.array(
        list(dozen_lags) + list(color_lags) + list(parity_lags) + list(high_low_lags)
    ).reshape(1, -1)  # Reshape to 1 row for prediction

    # Add a button for prediction
    if st.button("Get Prediction"):
        # Predict using the loaded model
        prediction = model.predict(input_features)

        # Extract predictions
        next_dozen_1, next_dozen_2, next_color = prediction[0]
        dozen_mapping_reverse = {0: "Dozen 1 (1-12)", 1: "Dozen 2 (13-24)", 2: "Dozen 3 (25-36)"}
        color_mapping_reverse = {0: "Black", 1: "Red", 2: "Green"}

        # Function to ensure valid dozen values (0, 1, or 2)
        def get_valid_dozen_value(value):
            # Ensure the value is within the valid range of 0, 1, or 2
            return value if value in [0, 1, 2] else 0  # Default to 0 (Dozen 1)

        # Ensure that the predicted dozen values are valid
        next_dozen_1 = get_valid_dozen_value(int(next_dozen_1))
        next_dozen_2 = get_valid_dozen_value(int(next_dozen_2))

        # Ensure that the two dozens are different
        if next_dozen_1 == next_dozen_2:
            next_dozen_2 = (next_dozen_2 + 1) % 3  # Shift the second dozen cyclically

        # Display predictions
        st.subheader("Prediction Results")
        st.write(f"Next Possible Dozen 1: {dozen_mapping_reverse[next_dozen_1]}")
        st.write(f"Next Possible Dozen 2: {dozen_mapping_reverse[next_dozen_2]}")
        st.write(f"Next Predicted Color: {color_mapping_reverse[next_color]}")

else:
    st.error("Could not retrieve spins. Please check the API or try again later.")


# In[ ]:




