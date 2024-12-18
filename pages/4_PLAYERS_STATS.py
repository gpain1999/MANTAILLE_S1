import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from PIL import Image
import math
import matplotlib.pyplot as plt

st.set_page_config(page_title="PLAYERS STATS", layout="wide")


data_dir = os.path.join(os.path.dirname(__file__), '../data')
# sys.path.append(os.path.join(os.path.dirname(__file__), './fonctions/fonctions'))
images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')  # Path to the images directory

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


selected_stats = st.sidebar.selectbox("Stat Selected", options=["SHOOTS","OTHERS","ALL"], index=0)

MODE = st.sidebar.selectbox("Aggregation method", options=["CUMULATED", "AVERAGE"], index=1)



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
        'Matchs joués': group.shape[0],
        'TIME': (group['Time ON'].agg(aggregation_func)).round(1),
        'Steals': (group['Steals'].agg(aggregation_func)).round(1),
        'Turn.': (group['Turn.'].agg(aggregation_func)).round(1),
        'Assists': (group['Assists'].agg(aggregation_func)).round(1),
        'Blocks': (group['Blocks'].agg(aggregation_func)).round(1),
        'Reb.Off.': (group['Reb.Off.'].agg(aggregation_func)).round(1),
        'Reb.Def.': (group['Reb.Def.'].agg(aggregation_func)).round(1),
        'Reb.Tot.': (group['Reb.Tot.'].agg(aggregation_func)).round(1),
        'PTS': (group['PTS'].agg(aggregation_func)).round(1),
        'PER': (group['PER'].agg(aggregation_func)).round(1),
        'I_PER': ((group['PER'].sum() / group['Time ON'].sum()**0.5) if aggregation_type == "CUMULATED" else (group['PER'].agg(aggregation_func) / group['Time ON'].agg(aggregation_func)**0.5)).round(1),
        '1M': (group['1PTS M'].agg(aggregation_func)).round(1),
        '1T': (group['1PTS T'].agg(aggregation_func)).round(1),
        'PP2FT': ((group['1PTS M'].agg(aggregation_func) * 2) / group['1PTS T'].agg(aggregation_func)).round(2),
        '2M': (group['2PTS M'].agg(aggregation_func)).round(1),
        '2T': (group['2PTS T'].agg(aggregation_func)).round(1),
        'PPS2': ((group['2PTS M'].agg(aggregation_func) * 2) / group['2PTS T'].agg(aggregation_func)).round(2),
        '3M': (group['3PTS M'].agg(aggregation_func)).round(1),
        '3T': (group['3PTS T'].agg(aggregation_func)).round(1),
        'PPS3': ((group['3PTS M'].agg(aggregation_func) * 3) / group['3PTS T'].agg(aggregation_func)).round(2),
        'NB FT': (group['1PTS T'].agg(aggregation_func)).round(1),
        'NB TIRS': ((group['2PTS T'].agg(aggregation_func) + group['3PTS T'].agg(aggregation_func))).round(1),
        'PPS': ((group['2PTS M'].agg(aggregation_func) * 2 + group['3PTS M'].agg(aggregation_func) * 3) /
                (group['2PTS T'].agg(aggregation_func) + group['3PTS T'].agg(aggregation_func))).round(2),
    }
    return pd.Series(result)

# Application de l'agrégation par groupe avec le paramètre choisi
aggregated_df = data.groupby('Nom').apply(lambda group: custom_aggregation(group, MODE))


aggregated_df['Matchs joués'] = aggregated_df['Matchs joués'].astype(int)

aggregated_df = aggregated_df.fillna(0)  # Remplacer les NaN par 0
aggregated_df = aggregated_df.reset_index()

# Ajouter une colonne temporaire pour calculer le produit
aggregated_df['Matchs x TIME'] = aggregated_df['Matchs joués'] * aggregated_df['TIME']

# Trier par cette colonne en ordre décroissant
aggregated_df = aggregated_df.sort_values(by='Matchs x TIME', ascending=False)

# Supprimer la colonne temporaire si elle n'est plus nécessaire
aggregated_df = aggregated_df.drop(columns=['Matchs x TIME']).reset_index(drop=True)

##################################################################



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
            <b>PLAYERS STATS</b>
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


if selected_stats == "SHOOTS" :
    aggregated_df_2 = aggregated_df[['Nom', 'Matchs joués', 'TIME','PER', 'I_PER','PTS','1M', '1T',
       'PP2FT', '2M', '2T', 'PPS2', '3M', '3T', 'PPS3', 'NB FT', 'NB TIRS',
       'PPS']]
elif selected_stats == "OTHERS" :
    aggregated_df_2 = aggregated_df[['Nom', 'Matchs joués', 'TIME','PER', 'I_PER','PTS','Reb.Off.', 'Reb.Def.', 'Reb.Tot.','Steals', 'Turn.', 'Assists', 'Blocks']]

else :
    aggregated_df_2 = aggregated_df[['Nom', 'Matchs joués', 'TIME','PER', 'I_PER','PTS','1M', '1T',
       'PP2FT', '2M', '2T', 'PPS2', '3M', '3T', 'PPS3', 'NB FT', 'NB TIRS',
       'PPS','Reb.Off.', 'Reb.Def.', 'Reb.Tot.','Steals', 'Turn.', 'Assists', 'Blocks']]


st.markdown(
    f'''
    <p style="font-size:{int(25)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
        <b></b>
    </p>
    ''',
    unsafe_allow_html=True
)

st.dataframe(aggregated_df_2, height=min(35 + 35*len(aggregated_df),900),width=2000,hide_index=True)  # Augmenter la hauteur du tableau

