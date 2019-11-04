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


def recommendations_system_simple(wines, users, ratings, size):
    # Prepare dataframe
    df = pd.merge(wines, ratings)
    df = pd.merge(df, users)
    df.drop(
        columns=['type', 'country', 'region', 'alcohol_content', 'producer', 'service', 'volume', 'grape', 'harvest',
                 'harmonization', 'image', 'user_name', 'gender', 'profession', 'age'], axis=1, inplace=True)

    # Set wine's ratings and total ratings.
    wines_sizes_ratings = df.groupby('wine_id').agg({'rating': [np.size, np.mean]})
    sizes = wines_sizes_ratings['rating']['size']
    means = wines_sizes_ratings['rating']['mean']
    df.drop_duplicates('wine_id', inplace=True)
    for i in range(len(wines_sizes_ratings)):
        df['vote_count'] = sizes.values
        df['vote_average'] = means.values

    # Displaying the ordered dataframe
    df.drop(columns=['user_id', 'rating'], axis=1, inplace=True)
    df = df.sort_values(['vote_average', 'vote_count'], ascending=False)

    # Mean of average of the dataframe.
    c = df['vote_average'].mean()

    # Now we calculate the 'm' to execute algorithm. -> m: minimum number of votes required to be listed
    m = df['vote_count'].quantile(0.75)

    # Qualified wines, based on vote counts.
    q_wines = df.copy().loc[df['vote_count'] >= m]

    def weighted_rating(x, m=m, c=c):
        v = x['vote_count']
        R = x['vote_average']
        # Calculation based on the IMDB formula
        return (v / (v + m) * R) + (m / (m + v) * c)

    q_wines['score'] = q_wines.apply(weighted_rating, axis=1)
    result = q_wines.sort_values('score', ascending=False)

    return jsonify(json.loads(result[0:int(size)].to_json(orient='records')))


def recommendations_system_collaborative(wines, ratings, wine_key, size):
    # Merge wine dataset with matrix dataset
    df = pd.merge(wines, ratings)

    # Average wine ratings
    wines_df_stats = df.groupby('wine_name').agg({'rating': [np.size, np.mean]})

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
        wine = wines[wines['wine_name'] == wine_key].index.tolist()
        wine = wine[0]

        wines['similarity'] = matrix_df.iloc[wine]
        wines.drop_duplicates('wine_id')

        result = pd.DataFrame(wines.sort_values(['similarity'], ascending=False))
        try:
            size = size
        except TypeError:
            return "Size cannot be null"

        return jsonify(json.loads(result[0:int(size)].to_json(orient='records')))

    except:
        return "Wine not found"


# Get the encoding of the file.
@app.route('/v1/file/encoding', methods=['GET'])
def get_encoding_file():
    src = request.args.get('src')
    encoding = find_encoding(src)
    return jsonify(encoding)


# This method receives a CSV containing:
# IMPORTANT: the csv files need must be at the root of the project within the data directory with their names 'wines' and 'ratings' or else change the variables with path.
# size: length of list to returns.
# Returns the best wines with simple recommendation.
@app.route('/v1/recommendations/sample/wines/csv', methods=['GET'])
def get_recommendations_system_simple_by_csv():
    # Detect the enconde having in wines csv file.
    wines_csv_encoding = find_encoding('data/wines.csv')
    wines_col = ['wine_id', 'wine_name', 'type', 'country', 'region', 'alcohol_content', 'producer', 'service',
                 'volume', 'grape', 'harvest', 'harmonization', 'image']
    wines = pd.read_csv('data/wines.csv', sep=';', encoding=wines_csv_encoding, names=wines_col, engine='python')

    # Detect the enconde having in ratings csv file.
    ratings_csv_encoding = find_encoding('data/wines.csv')
    ratings_col = ['user_id', 'wine_id', 'rating']
    ratings = pd.read_csv('data/ratings.csv', sep=';', encoding=ratings_csv_encoding, names=ratings_col,
                          engine='python')

    # Detect the enconde having in users csv file.
    users_csv_encoding = find_encoding('data/users.csv')
    users_col = ["user_id", 'user_name', 'gender', 'profession', 'age']
    users = pd.read_csv('data/users.csv', sep=';', encoding=users_csv_encoding, names=users_col, engine='python')

    return recommendations_system_simple(wines, users, ratings, 10)


# This method receives a JSON containing:
# wine_key: Name of the wine that was accessed and will be worked on in the recommendation algorithm.
# wines: JSON dataset with Wines.
# ratings: JSON dataset with Ratings.
# size: length of list to returns.
# Returns the best wines with simple recommendation.
@app.route('/v1/recommendations/sample/wines/csv', methods=['GET'])
def get_recommendations_system_simple_by_json():
    # Load Wines df
    try:
        wines_json = json.dumps(request.get_json()['wines'])
        wines_df = pd.read_json(wines_json)
    except:
        return "Wines set cannot be works"

    # Load Users df
    try:
        users_json = json.dumps(request.get_json()['users'])
        users_df = pd.read_json(users_json)
    except:
        return "Users set cannot be works"

    # Load Ratings df
    try:
        ratings_json = json.dumps(request.get_json()['ratings'])
        ratings_df = pd.read_json(ratings_json)
    except:
        return "Ratings set cannot be works"

    return recommendations_system_simple(wines_df, users_df, ratings_df)


# This method receives a CSV containing:
# IMPORTANT: the csv files need must be at the root of the project within the data directory with their names 'wines' and 'ratings' or else change the variables with path.
# wine_key: Name of the wine that was accessed and will be worked on in the recommendation algorithm.
# size: length of list to returns.
# Returns a json with top(size) wines recommended.
@app.route('/v1/recommendations/user-collaborative/wines/csv', methods=['GET'])
def get_recommendations_system_collaborative_by_csv():
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

    # Load Wines df
    wines_csv_src = 'data/wines.csv'
    wines_csv_encoding = find_encoding(wines_csv_src)
    wines_col = ['wine_id', 'wine_name', 'type', 'country', 'region', 'alcohol_content', 'producer', 'service',
                 'volume', 'grape', 'harvest', 'harmonization', 'image']
    wines_df = pd.read_csv(wines_csv_src, sep=';', names=wines_col, encoding=wines_csv_encoding, engine='python')

    # Load Ratings df
    ratings_csv_src = 'data/ratings.csv'
    ratings_csv_encoding = find_encoding(ratings_csv_src)
    ratings_col = ['user_id', 'wine_id', 'rating']
    ratings_df = pd.read_csv(ratings_csv_src, sep=';', names=ratings_col, encoding=ratings_csv_encoding,
                             engine='python')

    return recommendations_system_collaborative(wines_df, ratings_df, wine_key, size)


# This method receives a JSON containing:
# wine_key: Name of the wine that was accessed and will be worked on in the recommendation algorithm.
# wines: JSON dataset with Wines.
# ratings: JSON dataset with Ratings.
# size: length of list to returns.
# Returns the best wines with collaborative recommendation.
@app.route('/v1/recommendations/user-collaborative/wines/json', methods=['POST'])
def get_recommendations_system_collaborative_by_json():
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

    # Load Wines df
    try:
        wines_json = json.dumps(request.get_json()['wines'])
        wines_df = pd.read_json(wines_json)
    except:
        return "Wines set cannot be works"

    # Load Rating df
    try:
        ratings_json = json.dumps(request.get_json()['ratings'])
        ratings_df = pd.read_json(ratings_json)
    except:
        return "Ratings set cannot be works"

    return recommendations_system_collaborative(wines_df, ratings_df, wine_key, size)


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 80

    app.run(host='127.0.0.1', port=port, debug=True)
