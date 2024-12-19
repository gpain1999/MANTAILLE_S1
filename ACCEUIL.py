import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from PIL import Image
import math
import matplotlib.pyplot as plt

from auth import require_authentication

require_authentication()
data_dir = os.path.join(os.path.dirname(__file__), '../data')
# sys.path.append(os.path.join(os.path.dirname(__file__), './fonctions/fonctions'))
images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')  # Path to the images directory

st.set_page_config(page_title="Mantaille NM3 -  Data", layout="wide")

ms_path = f"images/LOGO.png"  # Chemin vers l'image
nm3_path = f"images/NM3.png"  # Chemin vers l'image


col1, col2,col3 = st.columns([1,3,1])

with col1 : 


    try:
        image = Image.open(ms_path)
        # Redimensionner l'image (par exemple, largeur de 300 pixels)
        # Définir la hauteur maximale
        max_height = 300

        # Calculer la nouvelle largeur en maintenant le ratio d'aspect
        new_width = int(image.width * (max_height / image.height))

        # Redimensionner l'image
        image = image.resize((new_width, max_height))

        # Afficher l'image redimensionnée
        st.image(image)
    except FileNotFoundError:
        pass

with col2 :
    st.markdown(
        f'''
        <p style="font-size:{int(60)}px; text-align: center; padding: 10pxs;">
            <b>MANTAILLE DATA - by gpain1999</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

with col3 : 


    try:
        image = Image.open(nm3_path)
        # Redimensionner l'image (par exemple, largeur de 300 pixels)
        # Définir la hauteur maximale
        max_height = 300

        # Calculer la nouvelle largeur en maintenant le ratio d'aspect
        new_width = int(image.width * (max_height / image.height))

        # Redimensionner l'image
        image = image.resize((new_width, max_height))


        # Afficher l'image redimensionnée
        st.image(image)
    except FileNotFoundError:
        pass
