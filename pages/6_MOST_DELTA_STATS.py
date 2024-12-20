import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from PIL import Image
import math
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# Ajouter le chemin de la racine du projet pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auth import require_authentication

require_authentication()

data_dir = os.path.join(os.path.dirname(__file__), '../data')
# sys.path.append(os.path.join(os.path.dirname(__file__), './fonctions/fonctions'))
images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')  # Path to the images directory


st.set_page_config(page_title="DELTA STATS WIN/LOSS", layout="wide")

from streamlit_js_eval import streamlit_js_eval
page_width = streamlit_js_eval(js_expressions='window.innerWidth', key='WIDTH',  want_output = True,)

local_c1 = "#B91A1E"
local_c2 = "#FFFFFF"
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

selected_range = st.sidebar.slider(
    "Select a ROUND range:",
    min_value=1,
    max_value=data["ROUND"].max(),
    value=(1, data["ROUND"].max()),
    step=1
)

data = data[(data["ROUND"]>= selected_range[0])&(data["ROUND"]<= selected_range[1])]

data = data.drop(columns=["ROUND","Date","Adversaire"])
coefficients = dict(zip(per["STATS"].to_list(), per[PER_TYPE].to_list()))
data["PER"] = (sum(data[col] * coeff for col, coeff in coefficients.items())).round(1)
data["I_PER"] = data["PER"]  / data["Time ON"]**0.5 

def custom_aggregation(group):

    result = {
        'Time ON': (group['Time ON'].sum()).round(1),

    }
    return pd.Series(result)

# Application de l'agrégation par groupe avec le paramètre choisi
top8 = data.groupby(["Nom"]).apply(lambda group: custom_aggregation(group)).reset_index().sort_values(by = "Time ON",ascending = False).head(8)["Nom"].to_list()

data = data[data["Nom"].isin(top8)]

data.rename(columns={'2PTS T': 'NB 2P SH.','3PTS T': 'NB 3P SH.'}, inplace=True)


# Identifier les colonnes de statistiques
stats_columns = [
    'PER','Steals', 'Turn.', 'Assists', 'Blocks', 'Reb.Off.', 'Reb.Def.', 'Reb.Tot.',
    '1PTS M', '1PTS T', '2PTS M', 'NB 2P SH.', '3PTS M', 'NB 3P SH.', 'Time ON', 'PTS',
    '1PTS R', '2PTS R', '3PTS R'
]

# Transformation des données
df_melted = data.melt(id_vars=['Nom', 'WIN'], value_vars=stats_columns,
                    var_name='STATS', value_name='VALUE')

# Calcul des moyennes globales et par modalité de WIN
global_means = df_melted.groupby(['Nom', 'STATS'])['VALUE'].mean().reset_index()
global_means.rename(columns={'VALUE': 'MOYENNE_GLOBAL'}, inplace=True)

modal_means = df_melted.groupby(['Nom', 'STATS', 'WIN'])['VALUE'].mean().unstack(fill_value=0).reset_index()
modal_means.columns.name = None  # Remove the columns index name

# Fusionner les moyennes globales et par modalité
result = pd.merge(global_means, modal_means, on=['Nom', 'STATS'])



# Renommer les colonnes pour plus de clarté
result.rename(columns={
    'YES': 'MOYENNE_WIN',
    'NO': 'MOYENNE_LOSS'
}, inplace=True)

result = result[[ "Nom" ,  "STATS" , "MOYENNE_GLOBAL" ,   "MOYENNE_WIN","MOYENNE_LOSS"]]
for p in top8 :
    pstats = (result[(result["Nom"]==p)&(result["STATS"].isin(['1PTS M', '1PTS T', '2PTS M', 'NB 2P SH.', '3PTS M', 'NB 3P SH.']))])

    #ligne 
    result.loc[len(result)] = [p,
                               "PPS",
                               (pstats[pstats["STATS"]=="2PTS M"]["MOYENNE_GLOBAL"].to_list()[0]*2 + pstats[pstats["STATS"]=="3PTS M"]["MOYENNE_GLOBAL"].to_list()[0]*3)/(pstats[pstats["STATS"]=="NB 2P SH."]["MOYENNE_GLOBAL"].to_list()[0] + pstats[pstats["STATS"]=="NB 3P SH."]["MOYENNE_GLOBAL"].to_list()[0]),
                               (pstats[pstats["STATS"]=="2PTS M"]["MOYENNE_WIN"].to_list()[0]*2 + pstats[pstats["STATS"]=="3PTS M"]["MOYENNE_WIN"].to_list()[0]*3)/(pstats[pstats["STATS"]=="NB 2P SH."]["MOYENNE_WIN"].to_list()[0] + pstats[pstats["STATS"]=="NB 3P SH."]["MOYENNE_GLOBAL"].to_list()[0]),
                               (pstats[pstats["STATS"]=="2PTS M"]["MOYENNE_LOSS"].to_list()[0]*2 + pstats[pstats["STATS"]=="3PTS M"]["MOYENNE_LOSS"].to_list()[0]*3)/(pstats[pstats["STATS"]=="NB 2P SH."]["MOYENNE_LOSS"].to_list()[0] + pstats[pstats["STATS"]=="NB 3P SH."]["MOYENNE_LOSS"].to_list()[0])
                               ]
    result.loc[len(result)] = [p,
                               "PP2S",
                               (pstats[pstats["STATS"]=="2PTS M"]["MOYENNE_GLOBAL"].to_list()[0]*2)/(pstats[pstats["STATS"]=="NB 2P SH."]["MOYENNE_GLOBAL"].to_list()[0]),
                               (pstats[pstats["STATS"]=="2PTS M"]["MOYENNE_WIN"].to_list()[0]*2)/(pstats[pstats["STATS"]=="NB 2P SH."]["MOYENNE_WIN"].to_list()[0]),
                               (pstats[pstats["STATS"]=="2PTS M"]["MOYENNE_LOSS"].to_list()[0]*2)/(pstats[pstats["STATS"]=="NB 2P SH."]["MOYENNE_LOSS"].to_list()[0]),
                               ]
    result.loc[len(result)] = [p,
                               "PP3S",
                               (pstats[pstats["STATS"]=="3PTS M"]["MOYENNE_GLOBAL"].to_list()[0]*3)/(pstats[pstats["STATS"]=="NB 3P SH."]["MOYENNE_GLOBAL"].to_list()[0]),
                               (pstats[pstats["STATS"]=="3PTS M"]["MOYENNE_WIN"].to_list()[0]*3)/(pstats[pstats["STATS"]=="NB 3P SH."]["MOYENNE_WIN"].to_list()[0]),
                               (pstats[pstats["STATS"]=="3PTS M"]["MOYENNE_LOSS"].to_list()[0]*3)/(pstats[pstats["STATS"]=="NB 3P SH."]["MOYENNE_LOSS"].to_list()[0]),
                               ]
    result.loc[len(result)] = [p,
                               "PP FT",
                               (pstats[pstats["STATS"]=="1PTS M"]["MOYENNE_GLOBAL"].to_list()[0])/(pstats[pstats["STATS"]=="1PTS T"]["MOYENNE_GLOBAL"].to_list()[0]),
                               (pstats[pstats["STATS"]=="1PTS M"]["MOYENNE_WIN"].to_list()[0])/(pstats[pstats["STATS"]=="1PTS T"]["MOYENNE_WIN"].to_list()[0]),
                               (pstats[pstats["STATS"]=="1PTS M"]["MOYENNE_LOSS"].to_list()[0])/(pstats[pstats["STATS"]=="1PTS T"]["MOYENNE_LOSS"].to_list()[0]),
                               ]
    result.loc[len(result)] = [p,
                               "NB SHOOT",
                               (pstats[pstats["STATS"]=="NB 3P SH."]["MOYENNE_GLOBAL"].to_list()[0])+(pstats[pstats["STATS"]=="NB 2P SH."]["MOYENNE_GLOBAL"].to_list()[0]),
                               (pstats[pstats["STATS"]=="3PTS M"]["MOYENNE_WIN"].to_list()[0])+(pstats[pstats["STATS"]=="NB 2P SH."]["MOYENNE_WIN"].to_list()[0]),
                               (pstats[pstats["STATS"]=="3PTS M"]["MOYENNE_LOSS"].to_list()[0])+(pstats[pstats["STATS"]=="NB 2P SH."]["MOYENNE_LOSS"].to_list()[0]),
                               ]
    result.loc[len(result)] = [p,
                               "NB FT",
                               (pstats[pstats["STATS"]=="1PTS T"]["MOYENNE_GLOBAL"].to_list()[0]),
                               (pstats[pstats["STATS"]=="1PTS T"]["MOYENNE_WIN"].to_list()[0]),
                               (pstats[pstats["STATS"]=="1PTS T"]["MOYENNE_LOSS"].to_list()[0]),
                               ]

result["DELTA"] = (result["MOYENNE_WIN"] - result["MOYENNE_LOSS"]).round(2)
result["DELTA %"] = (result["MOYENNE_WIN"] - result["MOYENNE_LOSS"]) /  result["MOYENNE_LOSS"]*100
result["DELTA %"] = result["DELTA %"].replace([np.inf, -np.inf], 0).fillna(0)
result["DELTA %"] = result["DELTA %"].astype(int)

result["MOYENNE_GLOBAL"] = (result["MOYENNE_GLOBAL"]).round(2)
result["MOYENNE_WIN"] = (result["MOYENNE_WIN"]).round(2)
result["MOYENNE_LOSS"] = (result["MOYENNE_LOSS"]).round(2)

result = result.reindex(result["DELTA %"].abs().sort_values(ascending=False).index)

result = result[(result["MOYENNE_WIN"]!=0)&(result["MOYENNE_LOSS"]!=0)]
result_global = result[result["STATS"].isin(['Reb.Off.','PER','Assists' , 'Reb.Def.','Steals' ,'Reb.Tot.' ,'Time ON' ,'Turn.' , 'Blocks'])]

result_adresse = result[result["STATS"].isin(['PP3S','PPS','PP2S','PP FT'])]
result_nb_shoot = result[result["STATS"].isin(['NB FT','NB 2P SH.','NB 3P SH.','NB SHOOT'])]

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
            <b>MOST DELTA STATS WIN/LOSS</b>
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

st.title(f"CLASSICS STATS DELTA : ")

NAME,STATS,Mean_Tot,Mean_WIN,Mean_LOSE,Delta,Delta_p,_,NAME2,STATS2,Mean_Tot2,Mean_WIN2,Mean_LOSE2,Delta2,Delta_p2 = st.columns([0.06,0.06,0.06,0.06,0.06,0.06,0.06,
                                                                                                                                0.06,
                                                                                                                                0.06,0.06,0.06,0.06,0.06,0.06,0.06,])
with NAME :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> PLAYERS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_global.head(10)["Nom"] :

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with STATS :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> STATS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_global.head(10)["STATS"] :

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Mean_Tot :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: black ;color: white; padding: 2px; border-radius: 5px;outline: 3px solid white;">
            <b> AVG TOT </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_global.head(10)["MOYENNE_GLOBAL"] :
        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: white ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )
with Mean_WIN :

    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: green ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> AVG WIN </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_global.head(10)["MOYENNE_WIN"] :
        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: #E8FFD9 ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Mean_LOSE :

    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: red ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> AVG LOSS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_global.head(10)["MOYENNE_LOSS"] :
        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: #FFD3D3 ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Delta :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> DELTA </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_global.head(10)["DELTA"] :
        if v > 0 :
            vf = f"+{v}"
            coloration = "green"
        elif v<0 :
            vf = v
            coloration = "red"
        else :
            vf = v
            coloration = "white"

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {coloration} ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {vf} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Delta_p :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> DELTA %</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_global.head(10)["DELTA %"] :
        if v > 0 :
            vf = f"+{v} % " 
            coloration = "green"
        elif v<0 :
            vf = f"{v} % " 
            coloration = "red"
        else :
            vf = f"{v} % " 
            coloration = "white"

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {coloration} ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {vf} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with NAME2 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> PLAYERS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_global.iloc[10:20]["Nom"] :

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with STATS2 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> STATS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_global.iloc[10:20]["STATS"] :

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Mean_Tot2 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: black ;color: white; padding: 2px; border-radius: 5px;outline: 3px solid white;">
            <b> AVG TOT </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_global.iloc[10:20]["MOYENNE_GLOBAL"] :
        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: white ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )
with Mean_WIN2 :

    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: green ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> AVG WIN </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_global.iloc[10:20]["MOYENNE_WIN"] :
        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: #E8FFD9 ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Mean_LOSE2 :

    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: red ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> AVG LOSS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_global.iloc[10:20]["MOYENNE_LOSS"] :
        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: #FFD3D3 ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Delta2 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> DELTA </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_global.iloc[10:20]["DELTA"] :
        if v > 0 :
            vf = f"+{v}"
            coloration = "green"
        elif v<0 :
            vf = v
            coloration = "red"
        else :
            vf = v
            coloration = "white"

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {coloration} ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {vf} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Delta_p2 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> DELTA %</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_global.iloc[10:20]["DELTA %"] :
        if v > 0 :
            vf = f"+{v} % " 
            coloration = "green"
        elif v<0 :
            vf = f"{v} % " 
            coloration = "red"
        else :
            vf = f"{v} % " 
            coloration = "white"

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {coloration} ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {vf} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

st.markdown(
    f'''
    <p style="font-size:{int(25)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
        <b></b>
    </p>
    ''',
    unsafe_allow_html=True
)

st.title(f"SHOOTING ACCURACY DELTA :")
NAME,STATS,Mean_Tot,Mean_WIN,Mean_LOSE,Delta,Delta_p,_,NAME2,STATS2,Mean_Tot2,Mean_WIN2,Mean_LOSE2,Delta2,Delta_p2 = st.columns([0.06,0.06,0.06,0.06,0.06,0.06,0.06,
                                                                                                                                0.06,
                                                                                                                                0.06,0.06,0.06,0.06,0.06,0.06,0.06,])
with NAME :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> PLAYERS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_adresse.head(10)["Nom"] :

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with STATS :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> STATS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_adresse.head(10)["STATS"] :

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Mean_Tot :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: black ;color: white; padding: 2px; border-radius: 5px;outline: 3px solid white;">
            <b> AVG TOT </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_adresse.head(10)["MOYENNE_GLOBAL"] :
        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: white ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )
with Mean_WIN :

    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: green ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> AVG WIN </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_adresse.head(10)["MOYENNE_WIN"] :
        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: #E8FFD9 ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Mean_LOSE :

    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: red ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> AVG LOSS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_adresse.head(10)["MOYENNE_LOSS"] :
        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: #FFD3D3 ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Delta :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> DELTA </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_adresse.head(10)["DELTA"] :
        if v > 0 :
            vf = f"+{v}"
            coloration = "green"
        elif v<0 :
            vf = v
            coloration = "red"
        else :
            vf = v
            coloration = "white"

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {coloration} ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {vf} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Delta_p :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> DELTA %</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_adresse.head(10)["DELTA %"] :
        if v > 0 :
            vf = f"+{v} % " 
            coloration = "green"
        elif v<0 :
            vf = f"{v} % " 
            coloration = "red"
        else :
            vf = f"{v} % " 
            coloration = "white"

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {coloration} ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {vf} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with NAME2 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> PLAYERS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_adresse.iloc[10:20]["Nom"] :

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with STATS2 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> STATS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_adresse.iloc[10:20]["STATS"] :

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Mean_Tot2 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: black ;color: white; padding: 2px; border-radius: 5px;outline: 3px solid white;">
            <b> AVG TOT </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_adresse.iloc[10:20]["MOYENNE_GLOBAL"] :
        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: white ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )
with Mean_WIN2 :

    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: green ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> AVG WIN </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_adresse.iloc[10:20]["MOYENNE_WIN"] :
        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: #E8FFD9 ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Mean_LOSE2 :

    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: red ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> AVG LOSS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_adresse.iloc[10:20]["MOYENNE_LOSS"] :
        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: #FFD3D3 ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Delta2 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> DELTA </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_adresse.iloc[10:20]["DELTA"] :
        if v > 0 :
            vf = f"+{v}"
            coloration = "green"
        elif v<0 :
            vf = v
            coloration = "red"
        else :
            vf = v
            coloration = "white"

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {coloration} ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {vf} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Delta_p2 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> DELTA %</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_adresse.iloc[10:20]["DELTA %"] :
        if v > 0 :
            vf = f"+{v} % " 
            coloration = "green"
        elif v<0 :
            vf = f"{v} % " 
            coloration = "red"
        else :
            vf = f"{v} % " 
            coloration = "white"

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {coloration} ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {vf} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )
st.markdown(
    f'''
    <p style="font-size:{int(25)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
        <b></b>
    </p>
    ''',
    unsafe_allow_html=True
)

st.title(f"SHOOTS SELECTION DELTA :")

NAME,STATS,Mean_Tot,Mean_WIN,Mean_LOSE,Delta,Delta_p,_,NAME2,STATS2,Mean_Tot2,Mean_WIN2,Mean_LOSE2,Delta2,Delta_p2 = st.columns([0.06,0.06,0.06,0.06,0.06,0.06,0.06,
                                                                                                                                0.06,
                                                                                                                                0.06,0.06,0.06,0.06,0.06,0.06,0.06,])
with NAME :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> PLAYERS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_nb_shoot.head(10)["Nom"] :

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with STATS :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> STATS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_nb_shoot.head(10)["STATS"] :

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Mean_Tot :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: black ;color: white; padding: 2px; border-radius: 5px;outline: 3px solid white;">
            <b> AVG TOT </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_nb_shoot.head(10)["MOYENNE_GLOBAL"] :
        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: white ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )
with Mean_WIN :

    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: green ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> AVG WIN </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_nb_shoot.head(10)["MOYENNE_WIN"] :
        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: #E8FFD9 ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Mean_LOSE :

    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: red ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> AVG LOSS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_nb_shoot.head(10)["MOYENNE_LOSS"] :
        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: #FFD3D3 ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Delta :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> DELTA </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_nb_shoot.head(10)["DELTA"] :
        if v > 0 :
            vf = f"+{v}"
            coloration = "green"
        elif v<0 :
            vf = v
            coloration = "red"
        else :
            vf = v
            coloration = "white"

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {coloration} ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {vf} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Delta_p :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> DELTA %</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_nb_shoot.head(10)["DELTA %"] :
        if v > 0 :
            vf = f"+{v} % " 
            coloration = "green"
        elif v<0 :
            vf = f"{v} % " 
            coloration = "red"
        else :
            vf = f"{v} % " 
            coloration = "white"

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {coloration} ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {vf} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with NAME2 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> PLAYERS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_nb_shoot.iloc[10:20]["Nom"] :

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with STATS2 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> STATS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_nb_shoot.iloc[10:20]["STATS"] :

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Mean_Tot2 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: black ;color: white; padding: 2px; border-radius: 5px;outline: 3px solid white;">
            <b> AVG TOT </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_nb_shoot.iloc[10:20]["MOYENNE_GLOBAL"] :
        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: white ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )
with Mean_WIN2 :

    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: green ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> AVG WIN </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_nb_shoot.iloc[10:20]["MOYENNE_WIN"] :
        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: #E8FFD9 ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Mean_LOSE2 :

    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: red ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> AVG LOSS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_nb_shoot.iloc[10:20]["MOYENNE_LOSS"] :
        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: #FFD3D3 ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Delta2 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> DELTA </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_nb_shoot.iloc[10:20]["DELTA"] :
        if v > 0 :
            vf = f"+{v}"
            coloration = "green"
        elif v<0 :
            vf = v
            coloration = "red"
        else :
            vf = v
            coloration = "white"

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {coloration} ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {vf} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Delta_p2 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> DELTA %</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_nb_shoot.iloc[10:20]["DELTA %"] :
        if v > 0 :
            vf = f"+{v} % " 
            coloration = "green"
        elif v<0 :
            vf = f"{v} % " 
            coloration = "red"
        else :
            vf = f"{v} % " 
            coloration = "white"

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {coloration} ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {vf} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )