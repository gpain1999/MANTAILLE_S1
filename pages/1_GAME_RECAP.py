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
# CSS pour masquer le footer et potentiellement d'autres éléments
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}  /* Cache le menu principal */
    footer {visibility: hidden;}    /* Cache le footer */
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

from streamlit_js_eval import streamlit_js_eval
page_width = streamlit_js_eval(js_expressions='window.innerWidth', key='WIDTH',  want_output = True,)


############################DATA ###########################

data = pd.read_csv(os.path.join(data_dir, f'all_boxescores.csv'))
game = pd.read_csv(os.path.join(data_dir, f'SCORE_GAME.csv'),sep=";")
per = pd.read_csv(os.path.join(data_dir, f'PER.csv'),sep=";")

game["GAME"] = game["DATE"]  +  "|"  + game["ADVERSAIRE"]
##################################################################

######################### PARAM



st.sidebar.header("SETTINGS")

G = st.sidebar.selectbox("SELECTED GAME", options=reversed(game["GAME"].to_list()), index=0)
PER_TYPE = st.sidebar.selectbox("SELECTED PER CALCUL", options=["GUITE PER","CLASSIC PER"], index=0)

coefficients = dict(zip(per["STATS"].to_list(), per[PER_TYPE].to_list()))


date, adv = G.split("|")

data_game = data[(data["Date"]==date)&(data["Adversaire"]==adv)].reset_index(drop = True)

data_game = data_game.drop(columns = ["Date","Adversaire"])

data_game =  data_game.sort_values(by = "Time ON",ascending = False).reset_index(drop = True)

# Calcul des totaux
totaux = data_game.drop(columns="Nom").sum()


data_game["PER"] = (sum(data_game[col] * coeff for col, coeff in coefficients.items())).round(1)

data_game["PPS"] = ((data_game["2PTS M"] * 2 + data_game["3PTS M"] *3) / (data_game["2PTS T"] + data_game["3PTS T"])).round(2)

PPS = ((data_game["2PTS M"].sum() * 2 + data_game["3PTS M"].sum() *3) / (data_game["2PTS T"].sum() + data_game["3PTS T"].sum())).round(2)

data_game = data_game.drop(columns = ["1PTS R","2PTS R", "3PTS R"])

data_game = data_game[["Nom","Time ON","PER","PTS",
                       "1PTS M", "1PTS T", "2PTS M", "2PTS T", "3PTS M", "3PTS T","PPS",
                       "Reb.Off.", "Reb.Def.", "Reb.Tot.",
                       "Steals", "Turn.", "Assists", "Blocks"]]
data_game = data_game.fillna(0)  # Remplacer les NaN par 0

game_game = game[(game["GAME"]==G)].reset_index(drop = True)

colonnes_a_convertir = [
    "PTS", "1PTS M", "1PTS T", "2PTS M", "2PTS T", 
    "3PTS M", "3PTS T", "Reb.Off.", "Reb.Def.", 
    "Reb.Tot.", "Steals", "Turn.", "Assists", "Blocks"
]

# Conversion des colonnes en int
data_game[colonnes_a_convertir] = data_game[colonnes_a_convertir].astype(int)

data_game.insert(2,"I_PER",(data_game["PER"]/data_game["Time ON"]**0.5).round(1))




if game_game["LIEU"].to_list()[0] == "DOM" :
    team_local = "MANTAILLE"
    LOCAL_SCORE = game_game["SCORE_MS"].to_list()[0]
    team_road = game_game["ADVERSAIRE"].to_list()[0]
    ROAD_SCORE = game_game["SCORE_ADV"].to_list()[0]
    local_c1 = "#B91A1E"
    local_c2 = "#FFFFFF"
else : 
    team_road = "MANTAILLE"
    ROAD_SCORE = game_game["SCORE_MS"].to_list()[0]
    team_local = game_game["ADVERSAIRE"].to_list()[0]
    LOCAL_SCORE = game_game["SCORE_ADV"].to_list()[0]
    local_c2 = "#B91A1E"
    local_c1 = "#FFFFFF"
#####################################################################


ms_path = f"images/LOGO.png"  # Chemin vers l'image
nm3_path = f"images/NM3.png"  # Chemin vers l'image


col1, com, col_adv, col_score1, col_score2 = st.columns([0.2, 0.1, 0.1, 0.35, 0.35])

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


with com : 

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





with col_score1 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
            <b></b>
        </p>
        ''',
        unsafe_allow_html=True
    )    
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.019)}px; text-align: center; background-color: {local_c1};color: {local_c2}; padding: 4px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b>{team_local} : {LOCAL_SCORE}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
with col_score2 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
            <b></b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.019)}px; text-align: center; background-color: {local_c2};color: {local_c1}; padding: 4px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b>{team_road} : {ROAD_SCORE}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

if (len(data_game)) == 0 :
    st.title(f"STATS DU MATCH NON DISPONIBLE POUR LE MOMENT.")
    st.stop()

st.title(f"STATS COLLECTIVES")

_,col_PPS,P1,P2,P3,col_reb,col_BC,col_st_as,_ = st.columns([0.045,0.13,0.13,0.13,0.13,0.13,0.13,0.13,0.045])

with col_PPS :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.013)}px; text-align: center; background-color: {local_c1};color: {local_c2}; padding: 4px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b>{PPS} PPS</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.013)}px; text-align: center; background-color: {local_c1};color: {local_c2}; padding: 4px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b>{round(data_game["PER"].sum(),1)} PER</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
with col_BC :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.013)}px; text-align: center; background-color: {local_c1};color: {local_c2}; padding: 4px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b>{data_game["Blocks"].sum()} BLOCKS</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.013)}px; text-align: center; background-color: {local_c1};color: {local_c2}; padding: 4px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b>{data_game["Turn."].sum()} TURN.</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
with P1 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.013)}px; text-align: center; background-color: {local_c1};color: {local_c2}; padding: 4px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b>{round(100*data_game["1PTS M"].sum()/data_game["1PTS T"].sum())}% LF</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.013)}px; text-align: center; background-color: {local_c1};color: {local_c2}; padding: 4px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b>{data_game["1PTS M"].sum()}/{data_game["1PTS T"].sum()}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
with P2 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.013)}px; text-align: center; background-color: {local_c1};color: {local_c2}; padding: 4px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b>{round(100*data_game["2PTS M"].sum()/data_game["2PTS T"].sum())}% 2PTS</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.013)}px; text-align: center; background-color: {local_c1};color: {local_c2}; padding: 4px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b>{data_game["2PTS M"].sum()}/{data_game["2PTS T"].sum()}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
with P3 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.013)}px; text-align: center; background-color: {local_c1};color: {local_c2}; padding: 4px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b>{round(100*data_game["3PTS M"].sum()/data_game["3PTS T"].sum())}% 3PTS</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.013)}px; text-align: center; background-color: {local_c1};color: {local_c2}; padding: 4px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b>{data_game["3PTS M"].sum()}/{data_game["3PTS T"].sum()}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
with col_reb :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.013)}px; text-align: center; background-color: {local_c1};color: {local_c2}; padding: 4px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b>{data_game["Reb.Off."].sum()} RO | {data_game["Reb.Def."].sum()} RD </b>
        </p>
        ''',
        unsafe_allow_html=True
    ) 
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.013)}px; text-align: center; background-color: {local_c1};color: {local_c2}; padding: 4px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b>{data_game["Reb.Tot."].sum()} REB  </b>
        </p>
        ''',
        unsafe_allow_html=True
    ) 
with col_st_as :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.013)}px; text-align: center; background-color: {local_c1};color: {local_c2}; padding: 4px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b>{data_game["Assists"].sum()} ASSISTS</b>
        </p>
        ''',
        unsafe_allow_html=True
    ) 
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.013)}px; text-align: center; background-color: {local_c1};color: {local_c2}; padding: 4px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b>{data_game["Steals"].sum()} STEALS</b>
        </p>
        ''',
        unsafe_allow_html=True
    )   

st.title(f"STATS INDIVIDUELLES")

cols = st.columns([1.5] + [1] * (len(data_game.columns)-1))  # Créer des colonnes dynamiques

for i, col in enumerate(cols):  # Itérer sur chaque colonne
    if i == 0 :

        col.markdown(
            f'''
            <p style="font-size:{int(page_width*0.008)}px; text-align: center; background-color: #B91A1E;color:#FFFFFF; padding: 2px; border-radius: 5px;outline: 3px solid #FFFFFF;">
                <b>{data_game.columns[i]}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        col.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
                <b></b>
            </p>
            ''',
            unsafe_allow_html=True
        )

        for v in data_game[data_game.columns[i]]:
            col.markdown(
                f'''
                <p style="font-size:{int(page_width*0.008)}px; text-align: center; background-color: #B91A1E;color: #FFFFFF; padding: 2px; border-radius: 5px;outline: 3px solid #FFFFFF;">
                    <b>{v}</b>
                </p>
                ''',
                unsafe_allow_html=True
            )

    else :
        col.markdown(
            f'''
            <p style="font-size:{int(page_width*0.008)}px; text-align: center; background-color: #B91A1E;color:#FFFFFF; padding: 2px; border-radius: 5px;outline: 3px solid #FFFFFF;">
                <b>{data_game.columns[i]}</b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        col.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
                <b></b>
            </p>
            ''',
            unsafe_allow_html=True
        )
        for v in data_game[data_game.columns[i]]:
            if (((v == data_game[data_game.columns[i]].max()) and (data_game.columns[i]!="Turn.")) or ((v == data_game[data_game.columns[i]].min()) and (data_game.columns[i]=="Turn."))) and (data_game[data_game.columns[i]].max() != data_game[data_game.columns[i]].min()) :
                col.markdown(
                    f'''
                    <p style="font-size:{int(page_width*0.008)}px; text-align: center; background-color: gold;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                        <b>{v}</b>
                    </p>
                    ''',
                    unsafe_allow_html=True)
            elif ((v == data_game[data_game.columns[i]].min()) or ((v == data_game[data_game.columns[i]].max()) and (data_game.columns[i]=="Turn."))) and (data_game[data_game.columns[i]].max() != data_game[data_game.columns[i]].min()) :
                col.markdown(
                    f'''
                    <p style="font-size:{int(page_width*0.008)}px; text-align: center; background-color: #00FFFF;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                        <b>{v}</b>
                    </p>
                    ''',
                    unsafe_allow_html=True)
            else :
                col.markdown(
                    f'''
                    <p style="font-size:{int(page_width*0.008)}px; text-align: center; background-color: #FFFFFF;color: #B91A1E; padding: 2px; border-radius: 5px;outline: 3px solid #B91A1E;">
                        <b>{v}</b>
                    </p>
                    ''',
                    unsafe_allow_html=True)


st.write(page_width)