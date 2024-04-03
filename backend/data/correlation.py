import pandas as pd
import json


def get_correlation():
    with open('backend/data/correlation_matrix.json', 'r') as file:
        correlation = json.load(file)

    columns = [item[0] for item in correlation['columns']]
    index = [item[0] for item in correlation['index']]
    df = pd.DataFrame(correlation['data'], columns=columns, index=index)

    return df
