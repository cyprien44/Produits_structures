import pandas as pd
import json


def get_correlation(selected_tickers):
    # Charger la matrice de corrélation complète à partir du fichier JSON
    with open('backend/data/correlation_matrix.json', 'r') as file:
        correlation_data = json.load(file)

    # Extraire les colonnes et les index à partir des données chargées
    columns = [item[0] for item in correlation_data['columns']]
    index = [item[0] for item in correlation_data['index']]
    
    # Créer un DataFrame à partir des données chargées
    full_correlation_matrix = pd.DataFrame(correlation_data['data'], columns=columns, index=index)

    # Filtrer la matrice de corrélation pour ne conserver que les colonnes et les lignes correspondant aux tickers sélectionnés
    filtered_correlation_matrix = full_correlation_matrix.loc[selected_tickers, selected_tickers]

    return filtered_correlation_matrix
