import os
import pandas as pd
from datetime import datetime

import os
import pandas as pd
from datetime import datetime

# Chemin du dossier contenant les fichiers .xlsx
folder_path = 'data/BOXESCORE/'

# Liste pour stocker tous les DataFrames
all_data = []

# Boucle à travers tous les fichiers dans le dossier
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        # Construction du chemin complet du fichier
        file_path = os.path.join(folder_path, file_name)
        
        # Lecture du fichier Excel
        df = pd.read_excel(file_path)

        # Renommer les colonnes
        df.columns = [
            "Nom", "Steals", "Turn.", "Assists", "Blocks", 
            "Reb.Off.", "Reb.Def.", "Reb.Tot.", 
            "1PTS M", "1PTS T", '1pts %', 
            "2PTS M", "2PTS T", '2pts %', 
            "3PTS M", "3PTS T", '3pts %', 
            "Marqués", "Tentatives", 
            "Time ON", "Points", "EVAL"
        ]
        
        # Filtrer les données
        df = df[df['Nom'].notna()]  # Retirer les lignes avec 'Nom' NaN
        df = df[df['Time ON'].notna()]  # Retirer les lignes avec 'Temps de jeu' NaN
        df = df[df['Nom'] != "TOTAL"].reset_index(drop=True)  # Retirer la ligne "TOTAL"
        df = df.drop(columns=['1pts %', '2pts %', '3pts %', "Marqués", "Tentatives", "Points", "EVAL"])  # Supprimer des colonnes inutiles
        df = df.fillna(0)  # Remplacer les NaN par 0
        df['Time ON'] = df['Time ON'].apply(lambda x: x.minute + x.second / 60).round(2)  # Convertir "Temps de jeu" en minutes
        
        # Extraction de la date et de l'adversaire depuis le nom du fichier
        date_str = file_name.split('_')[0]  # Première partie (avant '_')
        adversaire = file_name.split('_')[1].replace('.xlsx', '')  # Deuxième partie (après '_', sans '.xlsx')

        # Conversion de la date en format lisible (AAAA-MM-DD)
        date_obj = datetime.strptime(date_str, "%Y%m%d").date()

        # Ajout des nouvelles colonnes 'Date' et 'Adversaire'
        df.insert(0, 'Date', date_obj)         # Insérer la colonne 'Date' en première position (index 0)
        df.insert(1, 'Adversaire', adversaire) # Insérer la colonne 'Adversaire' en deuxième position (index 1)
        df["PTS"] = df["1PTS M"]*1 + df["2PTS M"]*2  + df["3PTS M"]*3
        df["1PTS R"] = df["1PTS T"] - df["1PTS M"]
        df["2PTS R"] = df["2PTS T"] - df["2PTS M"]
        df["3PTS R"] = df["3PTS T"] - df["3PTS M"]
        # Ajouter ce DataFrame à la liste
        all_data.append(df)

# Concaténer tous les DataFrames de la liste en un grand DataFrame
grand_df = pd.concat(all_data, ignore_index=True)

grand_df.to_csv("data/all_boxescores.csv",index=False)