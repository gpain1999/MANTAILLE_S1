import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from PIL import Image
import math
import matplotlib.pyplot as plt

# Ajouter le chemin de la racine du projet pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auth import require_authentication

require_authentication()
data_dir = os.path.join(os.path.dirname(__file__), '../data')
# sys.path.append(os.path.join(os.path.dirname(__file__), './fonctions/fonctions'))
images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')  # Path to the images directory



############################DATA ###########################
st.set_page_config(page_title="TEAM STATS", layout="wide")

from streamlit_js_eval import streamlit_js_eval
page_width = streamlit_js_eval(js_expressions='window.innerWidth', key='WIDTH',  want_output = True,)


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
data["TOTAL"] = "TOTAL"

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
        'NB TIRS': ((group['2PTS T'].agg(aggregation_func) + group['3PTS T'].agg(aggregation_func))).round(1),
        'PPS': ((group['2PTS M'].agg(aggregation_func) * 2 + group['3PTS M'].agg(aggregation_func) * 3) /
                (group['2PTS T'].agg(aggregation_func) + group['3PTS T'].agg(aggregation_func))).round(2),
    }
    return pd.Series(result)

# Application de l'agrégation par groupe avec le paramètre choisi
aggregated_df = data.groupby(["WIN","Date"]).apply(lambda group: custom_aggregation(group, "CUMULATED"))
aggregated_df_tot = data.groupby(["TOTAL","Date"]).apply(lambda group: custom_aggregation(group, "CUMULATED"))



aggregated_df = aggregated_df.reset_index()
aggregated_df_tot = aggregated_df_tot.reset_index()

def custom_aggregation2(group, aggregation_type="AVERAGE"):
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
        '1M': (group['1M'].agg(aggregation_func)).round(1),
        '1T': (group['1T'].agg(aggregation_func)).round(1),
        'PP2FT': ((group['1M'].agg(aggregation_func) * 2) / group['1T'].agg(aggregation_func)).round(2),
        '2M': (group['2M'].agg(aggregation_func)).round(1),
        '2T': (group['2T'].agg(aggregation_func)).round(1),
        'PPS2': ((group['2M'].agg(aggregation_func) * 2) / group['2T'].agg(aggregation_func)).round(2),
        '3M': (group['3M'].agg(aggregation_func)).round(1),
        '3T': (group['3T'].agg(aggregation_func)).round(1),
        'PPS3': ((group['3M'].agg(aggregation_func) * 3) / group['3T'].agg(aggregation_func)).round(2),
        'NB TIRS': ((group['3T'].agg(aggregation_func) + group['3M'].agg(aggregation_func))).round(1),
        'PPS': ((group['2M'].agg(aggregation_func) * 2 + group['3M'].agg(aggregation_func) * 3) /
                (group['2T'].agg(aggregation_func) + group['3T'].agg(aggregation_func))).round(2),
    }
    return pd.Series(result)

aggregated_df = aggregated_df.groupby(["WIN"]).apply(lambda group: custom_aggregation2(group, "AVERAGE"))
aggregated_df_tot = aggregated_df_tot.groupby(["TOTAL"]).apply(lambda group: custom_aggregation2(group, "AVERAGE"))


result = pd.concat([aggregated_df, aggregated_df_tot], axis=0)

result = result.T
result.index.name = "STATS"

result = (result).reset_index()


result["Delta"] = result["YES"] - result["NO"]




filtered_stats = ['PTS', 'PER','Steals', 'Turn.', 'Assists', 'Blocks', 'Reb.Off.',
 'Reb.Def.', 'Reb.Tot.', 'NB TIRS']

result_df1 = result[result['STATS'].isin(filtered_stats)]
result_df1['STATS'] = pd.Categorical(result_df1['STATS'], categories=filtered_stats, ordered=True)
result_df1 = result_df1.sort_values('STATS')

filtered_stats = ['1M',
 '1T', 'PP2FT', '2M', '2T', 'PPS2',
 '3M', '3T', 'PPS3','PPS']

result_df2 = result[result['STATS'].isin(filtered_stats)]
result_df2['STATS'] = pd.Categorical(result_df2['STATS'], categories=filtered_stats, ordered=True)
result_df2 = result_df2.sort_values('STATS')


ms_path = f"images/LOGO.png"  # Chemin vers l'image
nm3_path = f"images/NM3.png"  # Chemin vers l'image


local_c1 = "#B91A1E"
local_c2 = "#FFFFFF"
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
        <p style="font-size:{int(page_width*0.03)}px; text-align: left; padding: 10pxs;">
            <b>TEAM STATS </b>
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
    <p style="font-size:{int(page_width*0.013)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
        <b></b>
    </p>
    ''',
    unsafe_allow_html=True
)

st.title(f"STATS VARIATION ")

STATS,Mean_tot,Mean_WIN,Mean_LOSE,Delta,_,STATS2,Mean_tot2,Mean_WIN2,Mean_LOSE2,Delta2 = st.columns([0.126,0.081,0.081,0.081,0.081,0.1,0.126,0.081,0.081,0.081,0.081])

with STATS :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c2} ;color: {local_c1}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c1};">
            <b> STATS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_df1["STATS"] :

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Mean_tot :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: #D3D3D3  ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> AVG GLOBAL </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_df1['TOTAL'] :
        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: #C0C0C0 ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
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
    for v in result_df1['YES'] :
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
            <b> AVG LOSE </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_df1['NO'] :
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
    for i,v in enumerate(result_df1['Delta']) :
        if i != 3 :
            if v > 0 :
                vf = f"+{round(v,1)}"
                coloration = "green"
            elif v<0 :
                vf = round(v,1)
                coloration = "red"
            else :
                vf = round(v,1)
                coloration = "white"
        else :
            if v > 0 :
                vf = f"+{round(v,1)}"
                coloration = "red"
            elif v<0 :
                vf = round(v,1)
                coloration = "green"
            else :
                vf = round(v,1)
                coloration = "white"            

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {coloration} ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {vf} </b>
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
    for v in result_df2["STATS"] :

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
                <b> {v} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )

with Mean_tot2 :
    st.markdown(
        f'''
        <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: #D3D3D3  ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> AVG GLOBAL </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_df2['TOTAL'] :
        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: #C0C0C0 ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
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
    for v in result_df2['YES'] :
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
            <b> AVG LOSE </b>
        </p>
        ''',
        unsafe_allow_html=True
    )
    for v in result_df2['NO'] :
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
    for v in result_df2['Delta'] :
        if v > 0 :
            vf = f"+{round(v,1)}"
            coloration = "green"
        elif v<0 :
            vf = round(v,1)
            coloration = "red"
        else :
            vf = round(v,1)
            coloration = "white"

        st.markdown(
            f'''
            <p style="font-size:{int(page_width*0.01)}px; text-align: center; background-color: {coloration} ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
                <b> {vf} </b>
            </p>
            ''',
            unsafe_allow_html=True
        )
