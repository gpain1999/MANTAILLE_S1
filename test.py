
import pandas as pd
import os
data_dir = os.path.join(os.path.dirname(__file__), 'data')

data = pd.read_csv("data/all_boxescores.csv")
per = pd.read_csv(os.path.join(data_dir, f'PER.csv'),sep=";")

PER_TYPE = "GUITE PER"
coefficients = dict(zip(per["STATS"].to_list(), per[PER_TYPE].to_list()))
data["PER"] = (sum(data[col] * coeff for col, coeff in coefficients.items())).round(1)
data["I_PER"] = data["PER"]  / data["Time ON"]**0.5 

def custom_aggregation(group):
    result = {
        'Matchs joués': group.shape[0],
        'TIME': (group['Time ON'].mean()).round(1),
        'Steals' : (group['Steals'].mean()).round(1),
        'Turn.' : (group['Turn.'].mean()).round(1),
        'Assists' : (group['Assists'].mean()).round(1),
        'Blocks' : (group['Blocks'].mean()).round(1),
        'Reb.Off.' : (group['Reb.Off.'].mean()).round(1),
        'Reb.Def.' : (group['Reb.Def.'].mean()).round(1),
        'Reb.Tot.' : (group['Reb.Tot.'].mean()).round(1),
        'PTS' : (group['PTS'].mean()).round(1),
        'PER' : (group['PER'].mean()).round(1),
        'I_PER' : ((group['PER'].sum()/group['Time ON'].sum()**0.5)).round(1),
        '1M' : (group['1PTS M'].mean()).round(1),
        '1T' : (group['1PTS T'].mean()).round(1),
        'PP2FT': ((group['1PTS M'].sum() * 2) / group['1PTS T'].sum()).round(2),
        '2M' : (group['2PTS M'].mean()).round(1),
        '2T' : (group['2PTS T'].mean()).round(1),
        'PPS2': ((group['2PTS M'].sum() * 2) / group['2PTS T'].sum()).round(2),
        '3M' : (group['3PTS M'].mean()).round(1),
        '3T' : (group['3PTS T'].mean()).round(1),
        'PPS3': ((group['3PTS M'].sum() * 3) / group['3PTS T'].sum()).round(2),
        'NB FT' : (group['1PTS T'].mean()).round(1),
        'NB TIRS' : ((group['2PTS T'].mean() + group['3PTS T'].mean())).round(1),
        'PPS': ((group['2PTS M'].sum() * 2 + group['3PTS M'].sum() * 3) / (group['2PTS T'].sum() + group['3PTS T'].sum())).round(2),

    }
    return pd.Series(result)

# Application de l'agrégation par groupe
aggregated_df = data.groupby('Nom').apply(custom_aggregation)

aggregated_df['Matchs joués'] = aggregated_df['Matchs joués'].astype(int)

aggregated_df = aggregated_df.fillna(0)  # Remplacer les NaN par 0
aggregated_df = aggregated_df.reset_index()

# Ajouter une colonne temporaire pour calculer le produit
aggregated_df['Matchs x TIME'] = aggregated_df['Matchs joués'] * aggregated_df['TIME']

# Trier par cette colonne en ordre décroissant
aggregated_df = aggregated_df.sort_values(by='Matchs x TIME', ascending=False)

# Supprimer la colonne temporaire si elle n'est plus nécessaire
aggregated_df = aggregated_df.drop(columns=['Matchs x TIME']).reset_index(drop=True)

# Affichage du résultat
print(aggregated_df)