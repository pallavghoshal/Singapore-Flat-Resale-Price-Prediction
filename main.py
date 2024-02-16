import streamlit as st
from streamlit_option_menu import option_menu
import json
import requests
import pandas as pd
# from geopy.distance import geodesic
# import statistics
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from PIL import Image

# -------------------------Reading the data on Lat and Long of all the MRT Stations in Singapore------------------------
# data = pd.read_csv('singapore.csv')
# mrt_location = pd.DataFrame(data)

# -------------------------------This is the configuration page for our Streamlit Application---------------------------
st.set_page_config(
    page_title="Singapore Resale Flat Prices Prediction",
    page_icon=Image.open("ICN.png"),
    layout="wide"
)

# -------------------------------This is the sidebar in a Streamlit application, helps in navigation--------------------
start_color = "#6495ED"  # Red
end_color = "#FF7518"    # Blue

selected = option_menu(menu_title=None,
                       options=["About Project", "Predictions"],
                       icons=["house", "gear"],
                       default_index=0,
                       orientation="horizontal",
                       styles={"nav-link": {"font-size": "35px", "text-align": "centre", "margin": "0px 0px 0px 0px", "--hover-color": "#ec4454", "transition": "color 0.3s ease, background-color 0.9s ease"},
                               "icon": {"font-size": "35px"},
                               "container": {"max-width": "6000px"},
                               "nav-link-selected": {"background-color": "#ec2b3b"}}
                       )

# st.image("img.png", )
# st.markdown("""
# <div style='text-align:left'>
#     <h1 style='color:#6495ED;'>Industrial Copper Modelling</h1>
# </div>
# """, unsafe_allow_html=True)
gradient_style = f"""
    background: linear-gradient(to right, {start_color}, {end_color});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;  /* Center text horizontally */
"""


def setting_bg():
    st.markdown(f""" 
    <style>
        .stApp {{
            background: url("https://cutewallpaper.org/22/plane-colour-background-wallpapers/189265759.jpg");
            background-size: cover;
            transition: background 0.5s ease;
        }}
        h1,h2,h3,h4,h5,h6 {{
            color: #f3f3f3;
            font-family: 'Roboto', sans-serif;
        }}
        .stButton>button {{
            color: #4e4376;
            background-color: #f3f3f3;
            transition: all 0.3s ease-in-out;
            font-color: #ec4454;
        }}
        .stButton>button:hover {{
            color: #ec4454;
            background-color: #2b5876;
            font-color: #000000;
        }}
        .stTextInput>div>div>input {{
            color: #4e4376;
            background-color: #f3f3f3;
        }}
    </style>
    """, unsafe_allow_html=True)


setting_bg()

# -----------------------------------------------About Project Section--------------------------------------------------
if selected == "About Project":
    st.image("img.png", )
    st.markdown("# :red[Singapore Resale Flat Prices Prediction]")
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown("### :red[Technologies :] Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building, Data Wrangling, "
                "Model Deployment")
    st.markdown("### :red[Overview :] This project aims to construct a machine learning model and implement "
                "it as a user-friendly online application in order to provide accurate predictions about the "
                "resale values of apartments in Singapore. This prediction model will be based on past transactions "
                "involving resale flats, and its goal is to aid both future buyers and sellers in evaluating the "
                "worth of a flat after it has been previously resold. Resale prices are influenced by a wide variety "
                "of criteria, including location, the kind of apartment, the total square footage, and the length "
                "of the lease. The provision of customers with an expected resale price based on these criteria is "
                "one of the ways in which a predictive model may assist in the overcoming of these obstacles.")
    st.markdown("### :red[Domain :] Real Estate")

# ------------------------------------------------Predictions Section---------------------------------------------------
if selected == "Predictions":
    st.markdown("# :red[Predicting Results based on Trained Models]")
    st.markdown(
        "### :white[Predicting Resale Price (Regression Task 57% Accuracy)]")

    try:
        with st.form("form1"):

            # -----New Data inputs from the user for predicting the resale price-----
            flat_type = st.number_input("Flat Type")
            st.write(
                '###### :green[Hint: Enter 0 for 1ROOM, 1 for 2ROOM, 2 for 3ROOM 3 for 4ROOM, 4 for 5ROOM, 6 for EXECUTIVE, 7 for MULTI GENERATION, 8 for MULTI-GENERATION]')
            # block = st.text_input("Block Number")
            floor_area_sqm = st.number_input(
                'Floor Area (Per Square Meter)')
            lease_commence_date = st.number_input('Lease Commence Date')
            # storey_range = st.text_input("Storey Range (Format: 'Value1' TO 'Value2')")

            # -----Submit Button for PREDICT RESALE PRICE-----
            submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")

            if submit_button is not None:
                with open(r"pk_file/model.pkl", 'rb') as file:
                    loaded_model = pickle.load(file)
                with open(r'pk_file/scaler.pkl', 'rb') as f:
                    scaler_loaded = pickle.load(f)

                # -----Sending the user enter values for prediction to our model-----
                new_sample = np.array(
                    [[flat_type, floor_area_sqm, lease_commence_date]])
                new_sample = scaler_loaded.transform(new_sample)
                new_pred = loaded_model.predict(new_sample)[0]
                st.write(
                    '## :red[Predicted resale price:] ', (new_pred))

            # print(lease_commence_date)

    except Exception as e:
        st.write(
            "Enter the above values to get the predicted resale price of the flat")
        st.write("Error:", e)
