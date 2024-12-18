import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from PIL import Image
import math
import matplotlib.pyplot as plt


data_dir = os.path.join(os.path.dirname(__file__), '../data')
# sys.path.append(os.path.join(os.path.dirname(__file__), './fonctions/fonctions'))
images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')  # Path to the images directory

############################DATA ###########################
st.set_page_config(page_title="TEAM STATS GAME BY GAME", layout="wide")

data = pd.read_csv(os.path.join(data_dir, f'all_boxescores.csv'))
game = pd.read_csv(os.path.join(data_dir, f'SCORE_GAME.csv'),sep=";")
per = pd.read_csv(os.path.join(data_dir, f'PER.csv'),sep=";")
data = pd.merge(
    data,
    game[["DATE","WIN","ROUND",'SCORE_MS', 'SCORE_ADV']],
    how="left",
    left_on=["Date"],
    right_on=["DATE"],
)


# Sidebar : Curseur pour sélectionner la plage de ROUND
st.sidebar.header("SETTINGS")

PER_TYPE = st.sidebar.selectbox("SELECTED PER CALCUL", options=["GUITE PER","CLASSIC PER"], index=0)

selected_range = st.sidebar.slider(
    "Select a ROUND range:",
    min_value=1,
    max_value=data["ROUND"].max(),
    value=(1, data["ROUND"].max()),
    step=1
)

data = data[(data["ROUND"]>= selected_range[0])&(data["ROUND"]<= selected_range[1])]

coefficients = dict(zip(per["STATS"].to_list(), per[PER_TYPE].to_list()))
data["PER"] = (sum(data[col] * coeff for col, coeff in coefficients.items())).round(1)
data["I_PER"] = data["PER"]  / data["Time ON"]**0.5 

def custom_aggregation(group, aggregation_type="AVERAGE"):
    if aggregation_type == "AVERAGE":
        aggregation_func = 'mean'
    elif aggregation_type == "CUMULATED":
        aggregation_func = 'sum'
    else:
        raise ValueError("L'argument 'aggregation_type' doit être 'AVERAGE' ou 'CUMULATED'.")

    result = {
        'Steals': (group['Steals'].agg(aggregation_func)).round(1),
        'Turn.': (group['Turn.'].agg(aggregation_func)).round(1),
        'Assists': (group['Assists'].agg(aggregation_func)).round(1),
        'Blocks': (group['Blocks'].agg(aggregation_func)).round(1),
        'Reb.Off.': (group['Reb.Off.'].agg(aggregation_func)).round(1),
        'Reb.Def.': (group['Reb.Def.'].agg(aggregation_func)).round(1),
        'Reb.Tot.': (group['Reb.Tot.'].agg(aggregation_func)).round(1),
        'PTS': (group['PTS'].agg(aggregation_func)).round(1),
        'PER': (group['PER'].agg(aggregation_func)).round(1),
        '1M': (group['1PTS M'].agg(aggregation_func)).round(1),
        '1T': (group['1PTS T'].agg(aggregation_func)).round(1),
        'PP2FT': ((group['1PTS M'].agg(aggregation_func) * 2) / group['1PTS T'].agg(aggregation_func)).round(2),
        '2M': (group['2PTS M'].agg(aggregation_func)).round(1),
        '2T': (group['2PTS T'].agg(aggregation_func)).round(1),
        'PPS2': ((group['2PTS M'].agg(aggregation_func) * 2) / group['2PTS T'].agg(aggregation_func)).round(2),
        '3M': (group['3PTS M'].agg(aggregation_func)).round(1),
        '3T': (group['3PTS T'].agg(aggregation_func)).round(1),
        'PPS3': ((group['3PTS M'].agg(aggregation_func) * 3) / group['3PTS T'].agg(aggregation_func)).round(2),
        'NB T': ((group['2PTS T'].agg(aggregation_func) + group['3PTS T'].agg(aggregation_func))).round(1),
        'PPS': ((group['2PTS M'].agg(aggregation_func) * 2 + group['3PTS M'].agg(aggregation_func) * 3) /
                (group['2PTS T'].agg(aggregation_func) + group['3PTS T'].agg(aggregation_func))).round(2),
    }
    return pd.Series(result)

# Application de l'agrégation par groupe avec le paramètre choisi
aggregated_df = data.groupby(['Date','Adversaire',"WIN",'SCORE_MS','SCORE_ADV']).apply(lambda group: custom_aggregation(group, "CUMULATED"))

for c in ['Steals', 'Turn.', 'Assists', 'Blocks','Reb.Off.', 'Reb.Def.', 'Reb.Tot.', 'PTS','1M', '1T','2M', '2T',  '3M', '3T',  'NB T'] :
    aggregated_df[c] = aggregated_df[c].astype(int)

##################################################################
aggregated_df = aggregated_df.reset_index()

ms_path = f"images/LOGO.png"  # Chemin vers l'image
nm3_path = f"images/NM3.png"  # Chemin vers l'image


col1,col3,col2 = st.columns([1,1,5])

with col1 : 


    try:
        image = Image.open(ms_path)
        # Redimensionner l'image (par exemple, largeur de 200 pixels)
        # Définir la hauteur maximale
        max_height = 200

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
        <p style="font-size:{int(60)}px; text-align: left; padding: 10pxs;">
            <b>TEAM STATS GAME BY GAME</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

with col3 : 


    try:
        image = Image.open(nm3_path)
        # Redimensionner l'image (par exemple, largeur de 200 pixels)
        # Définir la hauteur maximale
        max_height = 200

        # Calculer la nouvelle largeur en maintenant le ratio d'aspect
        new_width = int(image.width * (max_height / image.height))

        # Redimensionner l'image
        image = image.resize((new_width, max_height))


        # Afficher l'image redimensionnée
        st.image(image)
    except FileNotFoundError:
        pass

st.markdown(
    f'''
    <p style="font-size:{int(25)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
        <b></b>
    </p>
    ''',
    unsafe_allow_html=True
)
aggregated_df.rename(columns={'SCORE_MS': 'MS'}, inplace=True)
aggregated_df.rename(columns={'SCORE_ADV': 'ADV'}, inplace=True)
aggregated_df.rename(columns={'Reb.Off.': 'Reb.O.'}, inplace=True)
aggregated_df.rename(columns={'Reb.Def.': 'Reb.D.'}, inplace=True)
aggregated_df.rename(columns={'Reb.Tot.': 'Reb.'}, inplace=True)

st.dataframe(aggregated_df, height=min(35 + 35*len(aggregated_df),900),width=2000,hide_index=True)  # Augmenter la hauteur du tableau

cols = st.columns([1.8]*2 + [1] * (len(aggregated_df.columns)-2))  # Créer des colonnes dynamiques

for i, col in enumerate(cols):  # Itérer sur chaque colonne
    if (i <5)  :
    
        col.markdown(
            f'''
            <p style="font-size:{int(15)}px; text-align: center; background-color: #B91A1E;color:#FFFFFF; padding: 2px; border-radius: 5px;outline: 3px solid #FFFFFF;">
                <b>{aggregated_df.columns[i]}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        col.markdown(
            f'''
            <p style="font-size:{int(27)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
                <b></b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        for v,c in zip(aggregated_df[aggregated_df.columns[i]],aggregated_df["WIN"]):
            if c == "YES" :
                col_a = "green"
            else :
                col_a = "red"
            col.markdown(
                f'''
                <p style="font-size:{int(15)}px; text-align: center; background-color: {col_a};color: #FFFFFF; padding: 2px; border-radius: 5px;outline: 3px solid #FFFFFF;">
                    <b>{v}</b>
                </p>
                ''',
                unsafe_allow_html=True
            )

    else:
        col.markdown(
            f'''
            <p style="font-size:{int(15)}px; text-align: center; background-color: #B91A1E;color:#FFFFFF; padding: 2px; border-radius: 5px;outline: 3px solid #FFFFFF;">
                <b>{aggregated_df.columns[i]}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        col.markdown(
            f'''
            <p style="font-size:{int(27)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
                <b></b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        for v in aggregated_df[aggregated_df.columns[i]]:
            if v == aggregated_df[aggregated_df.columns[i]].max() :
                col.markdown(
                    f'''
                    <p style="font-size:{int(15)}px; text-align: center; background-color: gold;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                        <b>{v}</b>
                    </p>
                    ''',
                    unsafe_allow_html=True)
            else :
                col.markdown(
                    f'''
                    <p style="font-size:{int(15)}px; text-align: center; background-color: #FFFFFF;color: #B91A1E; padding: 2px; border-radius: 5px;outline: 3px solid #B91A1E;">
                        <b>{v}</b>
                    </p>
                    ''',
                    unsafe_allow_html=True)
