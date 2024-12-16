import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from PIL import Image
import math
import matplotlib.pyplot as plt
from datetime import datetime

def plot_semi_circular_chart(value, t, size=300, font_size=20,m=True):
    """
    Create a semi-circular chart that is perfectly square without unnecessary white space.
    """
    if m : 
        marg = size
    else :
        marg = 0
    # Validate input
    if not 0 <= value <= 1:
        raise ValueError("Value must be between 0 and 1.")

    # Define chart data for the semi-circle
    value_percentage = value * 100
    filled_portion = value * 180  # Degrees for the filled portion
    empty_portion = 180 - filled_portion  # Degrees for the empty portion

    # Add a transparent placeholder to simulate the bottom half of the circle
    placeholder = 180

    # Create the semi-circular plot using a Pie chart
    fig = go.Figure()

    # Add the green (filled) and red (empty) portions
    fig.add_trace(go.Pie(
        values=[filled_portion, empty_portion, placeholder],
        labels=["Rempli", "Vide", ""],  # Empty label for the placeholder
        marker=dict(colors=["#00ff00", "#ff0000", "rgba(0,0,0,0)"]),  # Transparent for placeholder
        textinfo="none",  # Hide text on the slices
        hole=0.5,
        direction="clockwise",
        sort=False,
        showlegend=False
    ))

    # Add the percentage text in the center
    fig.update_layout(
        autosize=True,
        annotations=[
            dict(
                text=f"{t} : <b>{round(value_percentage,1)}%</b>",
                x=0.5, y=0.2,  # Center the text in the top half
                font_size=font_size,
                showarrow=False
            )
        ],
        # Set layout to remove unnecessary spaces
        margin=dict(t=0, b=0, l=0, r=0),
        width=size,  # Ensure square dimensions
        height=size,  # Ensure square dimensions
    )

    # Restrict to the top half of the circle and align properly
    fig.update_traces(
        rotation=270, 
        pull=[0, 0, 0]
    )  # Rotate to make green start on the top left

    # Tighten the figure for perfect square export
    fig.update_layout(
        autosize=False,
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot area
    )

    return fig



data_dir = os.path.join(os.path.dirname(__file__), '../data')
# sys.path.append(os.path.join(os.path.dirname(__file__), './fonctions/fonctions'))
images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')  # Path to the images directory


st.set_page_config(page_title="GAME RECAP", layout="wide")



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

P = st.sidebar.selectbox("SELECTED PLAYER", options=sorted(data["Nom"].unique().tolist()), index=0)
PER_TYPE = st.sidebar.selectbox("SELECTED PER CALCUL", options=["GUITE PER","CLASSIC PER"], index=0)
selected_stats = st.sidebar.selectbox("Stat Selected", options=["PER","I_PER","PTS","Reb.Tot.","Assists","Steals"])



ms_path = f"images/LOGO.png"  # Chemin vers l'image
nm3_path = f"images/NM3.png"  # Chemin vers l'image
p_path = f"images/PLAYERS/{P}.png"  # Chemin vers l'image


coefficients = dict(zip(per["STATS"].to_list(), per[PER_TYPE].to_list()))
data["PER"] = (sum(data[col] * coeff for col, coeff in coefficients.items())).round(1)
data["I_PER"] = data["PER"]  / data["Time ON"]**0.5 

min_selected_stats = data[selected_stats].min()
max_selected_stats = data[selected_stats].max()

data_indiv = data[data["Nom"]==P].reset_index(drop = True)

def custom_aggregation(group):
    result = {
        'Matchs joués': group.shape[0],
        'Steals' : (group['Steals'].mean()).round(1),
        'Turn.' : (group['Turn.'].mean()).round(1),
        'Assists' : (group['Assists'].mean()).round(1),
        'Blocks' : (group['Blocks'].mean()).round(1),
        'Reb.Off.' : (group['Reb.Off.'].mean()).round(1),
        'Reb.Def.' : (group['Reb.Def.'].mean()).round(1),
        'Reb.Tot.' : (group['Reb.Tot.'].mean()).round(1),
        'PTS' : (group['PTS'].mean()).round(1),
        'PER' : (group['PER'].mean()).round(1),
        '1M' : (group['1PTS M'].mean()).round(1),
        '2M' : (group['2PTS M'].mean()).round(1),
        '3M' : (group['3PTS M'].mean()).round(1),
        '1T' : (group['1PTS T'].mean()).round(1),
        '2T' : (group['2PTS T'].mean()).round(1),
        '3T' : (group['3PTS T'].mean()).round(1),
        'I_PER' : ((group['PER'].sum()/group['Time ON'].sum()**0.5)).round(1),
        'PP2FT1': ((group['1PTS M'].sum() * 2) / group['1PTS T'].sum()).round(2),
        'NB FT' : group['1PTS T'].mean(),
        'PPS2': ((group['2PTS M'].sum() * 2) / group['2PTS T'].sum()).round(2),
        'PPS3': ((group['3PTS M'].sum() * 3) / group['3PTS T'].sum()).round(2),
        'PPS': ((group['2PTS M'].sum() * 2 + group['3PTS M'].sum() * 3) / (group['2PTS T'].sum() + group['3PTS T'].sum())).round(2),
        'NB TIRS' : (group['2PTS T'].mean() + group['3PTS T'].mean()),
        'TIME': (group['Time ON'].mean()).round(1)
        
    }
    return pd.Series(result)



# Application de l'agrégation par groupe
aggregated_df = data_indiv.groupby('Nom').apply(custom_aggregation)
aggregated_df['Matchs joués'] = aggregated_df['Matchs joués'].astype(int)
aggregated_df['NB TIRS'] = aggregated_df['NB TIRS'].astype(int)
aggregated_df['NB FT'] = aggregated_df['NB FT'].astype(int)
aggregated_df = aggregated_df.fillna(0)  # Remplacer les NaN par 0
avg_data = aggregated_df.loc[aggregated_df.index == P].reset_index(drop = True)


local_c1 = "#B91A1E"
local_c2 = "#FFFFFF"

#####################################################################
col1, com, col_p, col_name = st.columns([0.2, 0.1, 0.1, 0.7])

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

with col_p :
    try:
        image = Image.open(p_path)
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


st.title(f"AVERAGES {P}")

stat1,stat2,stat3,stat4,stat5,stat6,stat7,stat8,stat9,stat10,stat11,stat12 = st.columns([1 for i in range(12)])

with stat1 :
    
    st.markdown(
        f'''
        <p style="font-size:{int(25)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 9px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b> NB GAME</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        f'''
        <p style="font-size:{int(30)}px; text-align: center; background-color: white ;color: black; padding: 12px; border-radius: 5px;outline: 3px solid black;">
            <b> {avg_data["Matchs joués"].to_list()[0]}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )



with stat2 :
    
    st.markdown(
        f'''
        <p style="font-size:{int(25)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 9px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b> TIME</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        f'''
        <p style="font-size:{int(30)}px; text-align: center; background-color: white ;color: black; padding: 12px; border-radius: 5px;outline: 3px solid black;">
            <b> {avg_data["TIME"].to_list()[0]}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
with stat3 :
    
    st.markdown(
        f'''
        <p style="font-size:{int(25)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 9px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b>PER</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        f'''
        <p style="font-size:{int(30)}px; text-align: center; background-color: white ;color: black; padding: 12px; border-radius: 5px;outline: 3px solid black;">
            <b> {avg_data["PER"].to_list()[0]}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

with stat4 :
    
    st.markdown(
        f'''
        <p style="font-size:{int(25)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 9px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b>PTS</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        f'''
        <p style="font-size:{int(30)}px; text-align: center; background-color: white ;color: black; padding: 12px; border-radius: 5px;outline: 3px solid black;">
            <b> {avg_data["PTS"].to_list()[0]}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

with stat5 :
    
    st.markdown(
        f'''
        <p style="font-size:{int(25)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 9px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b>AST</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        f'''
        <p style="font-size:{int(30)}px; text-align: center; background-color: white ;color: black; padding: 12px; border-radius: 5px;outline: 3px solid black;">
            <b> {avg_data["Assists"].to_list()[0]}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
with stat6 :
    
    st.markdown(
        f'''
        <p style="font-size:{int(25)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 9px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b>REB</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        f'''
        <p style="font-size:{int(30)}px; text-align: center; background-color: white ;color: black; padding: 12px; border-radius: 5px;outline: 3px solid black;">
            <b> {avg_data["Reb.Tot."].to_list()[0]}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )
with stat7 :
    st.markdown(
        f'''
        <p style="font-size:{int(12)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b> DEF. REB. </b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        f'''
        <p style="font-size:{int(15)}px; text-align: center; background-color: white ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> {avg_data["Reb.Def."].to_list()[0]}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )


    st.markdown(
        f'''
        <p style="font-size:{int(12)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b> OFF. REB. </b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        f'''
        <p style="font-size:{int(15)}px; text-align: center; background-color: white ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> {avg_data["Reb.Off."].to_list()[0]}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

with stat8 :
    st.markdown(
        f'''
        <p style="font-size:{int(12)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b> I_PER </b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        f'''
        <p style="font-size:{int(15)}px; text-align: center; background-color: white ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> {avg_data["I_PER"].to_list()[0]}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )


    st.markdown(
        f'''
        <p style="font-size:{int(12)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b> %WIN </b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    print(data_indiv[["Time ON"]])
    st.markdown(
        f'''
        <p style="font-size:{int(15)}px; text-align: center; background-color: white ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> {round(100 * (data_indiv["WIN"] == "YES").mean())} %</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

with stat9 :
    st.markdown(
        f'''
        <p style="font-size:{int(12)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b> BLOCKS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        f'''
        <p style="font-size:{int(15)}px; text-align: center; background-color: white ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> {avg_data["Blocks"].to_list()[0]}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )


    st.markdown(
        f'''
        <p style="font-size:{int(12)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b> STEALS </b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        f'''
        <p style="font-size:{int(15)}px; text-align: center; background-color: white ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> {avg_data["Steals"].to_list()[0]}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

with stat10 :
    st.markdown(
        f'''
        <p style="font-size:{int(12)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b> FT MARQ. </b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        f'''
        <p style="font-size:{int(15)}px; text-align: center; background-color: white ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> {avg_data["1M"].to_list()[0]}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )


    st.markdown(
        f'''
        <p style="font-size:{int(12)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b> FT TENT. </b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        f'''
        <p style="font-size:{int(15)}px; text-align: center; background-color: white ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> {avg_data["1T"].to_list()[0]}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

with stat11 :
    st.markdown(
        f'''
        <p style="font-size:{int(12)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b> 2PTS MARQ. </b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        f'''
        <p style="font-size:{int(15)}px; text-align: center; background-color: white ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> {avg_data["2M"].to_list()[0]}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )


    st.markdown(
        f'''
        <p style="font-size:{int(12)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b> 2PTS TENT. </b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        f'''
        <p style="font-size:{int(15)}px; text-align: center; background-color: white ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> {avg_data["2T"].to_list()[0]}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )

with stat12 :
    st.markdown(
        f'''
        <p style="font-size:{int(12)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b> 3PTS MARQ. </b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        f'''
        <p style="font-size:{int(15)}px; text-align: center; background-color: white ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> {avg_data["3M"].to_list()[0]}</b>
        </p>
        ''',
        unsafe_allow_html=True
    )


    st.markdown(
        f'''
        <p style="font-size:{int(12)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b> 3PTS TENT. </b>
        </p>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        f'''
        <p style="font-size:{int(15)}px; text-align: center; background-color: white ;color: black; padding: 2px; border-radius: 5px;outline: 3px solid black;">
            <b> {avg_data["3T"].to_list()[0]}</b>
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
col1, col2, col3 = st.columns([0.2,0.4,0.4])


with col1:
    st.markdown(
        f'''
        <p style="font-size:{int(25)}px; text-align: center; background-color: {local_c1} ;color: {local_c2}; padding: 2px; border-radius: 5px;outline: 3px solid {local_c2};">
            <b>{avg_data["PPS"].to_list()[0]:.2f} PTS PER SHOOT</b>
        </p>
        ''',
        unsafe_allow_html=True
        )
    _, i, _ = st.columns([1.5,5.5,1])
    with i :


    
        fig2 = plot_semi_circular_chart(data_indiv["1PTS M"].sum()/data_indiv["1PTS T"].sum() if data_indiv["1PTS T"].sum() != 0 else 0,"FT",size=int(110),font_size=int(17),m=False)
        st.plotly_chart(fig2)
        fig2 = plot_semi_circular_chart(data_indiv["2PTS M"].sum()/data_indiv["2PTS T"].sum() if data_indiv["2PTS T"].sum() != 0 else 0,"2P",size=int(110),font_size=int(17),m=False)
        st.plotly_chart(fig2)
        fig2 = plot_semi_circular_chart(data_indiv["3PTS M"].sum()/data_indiv["3PTS T"].sum() if data_indiv["3PTS T"].sum() != 0 else 0,"3P",size=int(110),font_size=int(17),m=False)
        st.plotly_chart(fig2)

min_time_on = data_indiv["Time ON"].min()*0.9
max_time_on = data_indiv["Time ON"].max()*1.1

with col2:

    # Création du graphique avec la nouvelle courbe de moyenne glissante
    fig = go.Figure()

    # Barres pour les stats
    for i, row in data_indiv.iterrows():
        color = "green" if row["WIN"] == "YES" else "red"
        fig.add_trace(go.Bar(
            x=[row["ROUND"]],
            y=[row[selected_stats]],
            marker_color=color,
            yaxis="y1",
            showlegend=False
        ))

    # Courbe pour les minutes
    fig.add_trace(go.Scatter(
        x=data_indiv["ROUND"],
        y=data_indiv["Time ON"],
        mode="lines+markers",
        line=dict(color='blue', width=2),
        marker=dict(size=8, symbol="circle", color="blue"),
        yaxis="y2",
        name="Minutes played",
        showlegend=False

    ))


    # Mise à jour du layout
    fig.update_layout(
        autosize=True,
        title=f'{P} AVG: {avg_data[selected_stats].sum()} {selected_stats}  ON {avg_data["TIME"].sum()} MIN ',
        xaxis=dict(
            title="ROUND",
            showgrid=False,
            tickmode='linear',
        ),
        yaxis=dict(
            title=selected_stats,
            titlefont=dict(color="grey"),
            tickfont=dict(color="grey"),
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            range=[min_selected_stats, max_selected_stats],  # Fixation de la plage
        ),
        yaxis2=dict(
            title="Minutes",
            titlefont=dict(color="grey"),
            tickfont=dict(color="grey"),
            overlaying="y",
            side="right",
            showgrid=False,
            range=[min_time_on, max_time_on],  # Fixation de la plage
        ),
        legend=dict(
            orientation="h",  # Layout horizontal
            yanchor="top",
            y=-0.3,  # Position sous le graphique
            xanchor="center",
            x=0.5  # Centré horizontalement
        ),
        margin=dict(l=50, r=50, t=50, b=100),
        height=450,
        dragmode="pan"  
    )


    st.plotly_chart(fig, use_container_width=True,static_plot=True)


with col3:
    # Création des boxplots avec Plotly Graph Objects
    box_fig = go.Figure()

    # Boxplot global pour PER
    box_fig.add_trace(go.Box(
        y=data_indiv[selected_stats],
        name="Global",
        marker_color="blue"
    ))

    # Boxplots pour chaque modalité de WIN
    for win_status in reversed(sorted(data_indiv["WIN"].unique())):
        if win_status == "YES" :
            ecriture = "WIN"
        else :
            ecriture = "LOSE"

        box_fig.add_trace(go.Box(
            y=data_indiv[data_indiv["WIN"] == win_status][selected_stats],
            name=f"{ecriture}",
            marker_color="green" if win_status == "YES" else "red"
        ))

    # Configuration du layout des boxplots
    box_fig.update_layout(
        autosize=True,  # Le graphique s'ajuste automatiquement
        title=f"Distribution de {selected_stats} (Global et par résultat)",
        yaxis=dict(
            title=selected_stats,
            showgrid=True,
            range=[min_selected_stats, max_selected_stats]
        ),
        xaxis=dict(
            title="Catégories",
            showgrid=False
        ),
        margin=dict(l=50, r=50, t=50, b=50),
        height=450,
        showlegend=False,
        dragmode="pan"
    )

    
    # Affichage des boxplots
    st.plotly_chart(box_fig, use_container_width=True)

st.markdown(
    f'''
    <p style="font-size:{int(25)}px; text-align: center; background-color: grey;color: black; padding: 3px; border-radius: 5px;">
        <b></b>
    </p>
    ''',
    unsafe_allow_html=True
)


st.title(f"STATS DETAILS : {P}")

data_indiv2 = data_indiv.drop(columns = ["DATE","Nom"])
data_indiv2 = data_indiv2.sort_values(by = "Date",ascending = False)

data_indiv2 = data_indiv2[['Date', 'Adversaire', 'WIN',
                           'Time ON', 'PER', 'I_PER',
                           'PTS','1PTS M', '1PTS T', '2PTS M',
                           '2PTS T', '3PTS M', '3PTS T', 'Reb.Tot.',
                           'Reb.Off.', 'Reb.Def.','Steals', 'Turn.', 
                           'Assists', 'Blocks']]

data_indiv2['I_PER'] = (data_indiv2['I_PER']).round(1)
cols = st.columns([1.5]*2 + [1] * (len(data_indiv2.columns)-2))  # Créer des colonnes dynamiques

for i, col in enumerate(cols):  # Itérer sur chaque colonne
    if (i == 0) or (i==1) or (i==2)  :
    
        col.markdown(
            f'''
            <p style="font-size:{int(15)}px; text-align: center; background-color: #B91A1E;color:#FFFFFF; padding: 2px; border-radius: 5px;outline: 3px solid #FFFFFF;">
                <b>{data_indiv2.columns[i]}</b>
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

        for v,c in zip(data_indiv2[data_indiv2.columns[i]],data_indiv2["WIN"]):
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
                <b>{data_indiv2.columns[i]}</b>
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
        for v in data_indiv2[data_indiv2.columns[i]]:
            if v == data_indiv2[data_indiv2.columns[i]].max() :
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
#st.dataframe(data_indiv2, height=min(37*(len(data_indiv)+1),900),width=3000,hide_index=True)  
