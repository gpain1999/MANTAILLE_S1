
import pandas as pd

grand_df = pd.read_csv("data/all_boxescores.csv")
def custom_aggregation(group):
    result = {
        'PTS' : ((group['1PTS M'].mean() + group['2PTS M'].mean() *2 + group['3PTS M'].mean() *3)).round(1),
        'I_PTS' : ((group['1PTS M'].mean() + group['2PTS M'].mean() *2 + group['3PTS M'].mean() *3)/(group['Temps de jeu'].mean())**0.5).round(2),
        'PP2FT1': ((group['1PTS M'].sum() * 2) / group['1PTS T'].sum()).round(2),
        'NB FT' : group['1PTS T'].sum(),
        'PPS2': ((group['2PTS M'].sum() * 2) / group['2PTS T'].sum()).round(2),
        'PPS3': ((group['3PTS M'].sum() * 3) / group['3PTS T'].sum()).round(2),
        'PPS': ((group['2PTS M'].sum() * 2 + group['3PTS M'].sum() * 3) / (group['2PTS T'].sum() + group['3PTS T'].sum())).round(2),
        'NB TIRS' : (group['2PTS T'].sum() + group['3PTS T'].sum()),
        'Temps de jeu': group['Temps de jeu'].sum(),
        'TDJ Moyen': (group['Temps de jeu'].mean()).round(2),
        'Matchs joués': group.shape[0]
    }
    return pd.Series(result)

# Application de l'agrégation par groupe
aggregated_df = grand_df.groupby('Nom').apply(custom_aggregation)
aggregated_df['Matchs joués'] = aggregated_df['Matchs joués'].astype(int)
aggregated_df['NB TIRS'] = aggregated_df['NB TIRS'].astype(int)
aggregated_df['NB FT'] = aggregated_df['NB FT'].astype(int)

aggregated_df = aggregated_df.sort_values(by = "Temps de jeu",ascending = False)

# Affichage du résultat
print(aggregated_df)