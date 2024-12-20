import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from PIL import Image
import math
import matplotlib.pyplot as plt
from datetime import datetime
# Ajouter le chemin de la racine du projet pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auth import require_authentication

require_authentication()

data_dir = os.path.join(os.path.dirname(__file__), '../data')
# sys.path.append(os.path.join(os.path.dirname(__file__), './fonctions/fonctions'))
images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')  # Path to the images directory


st.set_page_config(page_title="GAME RECAP", layout="wide")

from streamlit_js_eval import streamlit_js_eval
page_width = streamlit_js_eval(js_expressions='window.innerWidth', key='WIDTH',  want_output = True,)


############################DATA ###########################

data = pd.read_csv(os.path.join(data_dir, f'all_boxescores.csv'))
game = pd.read_csv(os.path.join(data_dir, f'SCORE_GAME.csv'),sep=";")
per = pd.read_csv(os.path.join(data_dir, f'PER.csv'),sep=";")


data = pd.merge(
    data,
    game[["DATE","WIN","ROUND"]],
    how="left",
    left_on=["Date"],
    right_on=["DATE"],
)

##################################################################
######################### PARAM



# Sidebar : Curseur pour sélectionner la plage de ROUND
st.sidebar.header("SETTINGS")
PER_TYPE = st.sidebar.selectbox("SELECTED PER CALCUL", options=["GUITE PER","CLASSIC PER"], index=0)
