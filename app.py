import json
import sys

import chardet
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.metrics import pairwise_distances

app = Flask(__name__)


# Identify the encoding of the file.
def find_encoding(src):
    file = open(src, 'rb').read()
    result = chardet.detect(file)
    return result['encoding']


# Load dataset from CSV file.
# Return Dataframe.
def load_by_csv(src, cols, encoding):
    return pd.read_csv(src, sep=';', names=cols, encoding=encoding, engine='python')


def recommendations_system(wines, ratings, wine_key, size):
    # Merge wine dataset with matrix dataset
    df = pd.merge(wines, ratings)

    # Average wine ratings
    wines_df_stats = df.groupby('name').agg({'rating': [np.size, np.mean]})

    # Filters the statistical dataset with wines that have more than x analyzes.
    min_10 = wines_df_stats['rating']['size'] >= 3
    wines_df_stats[min_10].sort_values([('rating', 'mean')], ascending=False)

    # Pivo Table
    matrix_df = ratings.pivot_table(index=['wine_id'], columns=['user_id'], values=['rating']).reset_index(drop=True)
    matrix_df.fillna(0, inplace=True)

    # Calculate Cosine Similarity.
    wines_similarity = 1 - pairwise_distances(matrix_df.to_numpy(), metric='cosine')
    np.fill_diagonal(wines_similarity, 0)

    # Set Similarities to Matrix Dataframe.
    matrix_df = pd.DataFrame(wines_similarity)

    # Recommendation System
    try:
        wine = wines[wines['name'] == wine_key].index.tolist()
        wine = wine[0]

        wines['similarity'] = matrix_df.iloc[wine]
        wines.columns = ['wine_id', 'name', 'type', 'country', 'region', 'alcohol_content', 'producer', 'service',
                         'volume', 'vintage', 'similarity']

        result = pd.DataFrame(wines.sort_values(['similarity'], ascending=False))
        try:
            size = size
        except TypeError:
            return jsonify("Size cannot be null")

        return jsonify(json.loads(result[1: int(size)].to_json(orient='records')))

    except:
        return jsonify("Wine not found")


# Get the encoding of the file.
@app.route('/v1/file/encoding', methods=['GET'])
def get_encoding_file():
    src = request.args.get('src')
    encoding = find_encoding(src)
    return jsonify(encoding)


# This method receives a CSV containing:
# IMPORTANT: the csv files need must be at the root of the project within the data directory with their names 'wines' and 'ratings' or else change the variables with path.
# wine_key: Name of the wine that was accessed and will be worked on in the recommendation algorithm.
# size: length of list to returns.
# Returns a json with top(size) wines recommended.
@app.route('/v1/recommendations/wines/csv', methods=['GET'])
def get_recommendations_by_csv():
    # Key to works
    try:
        wine_key = request.args.get('wine_key')
    except KeyError:
        return "Wine Key cannot be null"

    # number of index to return in algorithm
    try:
        size = request.args.get('size')
    except KeyError:
        return "Size cannot be null"

    # Load Wines Dataframe
    wines_csv_src = 'data/wines.csv'
    wines_csv_encoding = find_encoding(wines_csv_src)
    wines_col = ['wine_id', 'name', 'type', 'country', 'region', 'alcohol_content', 'producer', 'service', 'volume',
                 'vintage']
    wines_df = pd.read_csv(wines_csv_src, sep=';', names=wines_col, encoding=wines_csv_encoding, engine='python')

    # Load Ratings Dataframe
    ratings_csv_src = 'data/ratings.csv'
    ratings_csv_encoding = find_encoding(ratings_csv_src)
    ratings_col = ['user_id', 'wine_id', 'rating']
    ratings_df = pd.read_csv(ratings_csv_src, sep=';', names=ratings_col, encoding=ratings_csv_encoding,
                             engine='python')

    return recommendations_system(wines_df, ratings_df, wine_key, size)


# This method receives a JSON containing:
# wine_key: Name of the wine that was accessed and will be worked on in the recommendation algorithm.
# wines: Example -> [{"wine_id": 1, "name": "Expedicion Single Vineyard Selection Cabernet Sauvignon 2019","type": "Tinto","country": "Chile","region": "Vale Central", "alcohol_content": 13, "producer": "Finca Patagonia", "service": 17, "volume": 750, "vintage": 2019}].
# ratings: Example -> [{"user_id": 1, "wine_id", "rating": 5}]
# size: length of list to returns.
# Returns a json with top(size) wines recommended.
@app.route('/v1/recommendations/wines/json', methods=['POST'])
def get_recommendations_by_json():
    # Key to works
    try:
        wine_key = request.get_json()['wine_key']
    except KeyError:
        return "Wine Key cannot be null"

    # number of index to return in algorithm
    try:
        size = request.get_json()['size']
    except KeyError:
        return "Size cannot be null"

    # Load Wines Dataframe
    try:
        wines_json = json.dumps(request.get_json()['wines'])
        wines_df = pd.read_json(wines_json)
    except:
        return "Wines set cannot be works"

    # Load Rating Dataframe
    try:
        ratings_json = json.dumps(request.get_json()['ratings'])
        ratings_df = pd.read_json(ratings_json)
    except:
        return "Ratings set cannot be works"

    return recommendations_system(wines_df, ratings_df, wine_key, size)


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 80

    app.run(host='127.0.0.1', port=port, debug=True)
