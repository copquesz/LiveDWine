import json
import sys

import numpy as np
import pandas as pd
import pymysql
from flask import Flask, request, jsonify
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)


# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

def create_soup(x):
    return ' '.join(x['type']) + ' ' + ' '.join(x['country']) + ' ' + x['region'] + ' ' + ' '.join(x['producer']) + ' ' + ' '.join(x['grape']) + ' ' + ' '.join(x['harmonization'])

# This method takes the database connection.
def get_database_connection():
    return pymysql.connect(host='localhost', port=int(3306), user='root', passwd='dev@1998', db="livedwine",
                           charset='utf8')


# This method closes the database connection.
def close_database_connection(conn):
    conn.close()


# Load dataframe with all wines, users and ratings.
def load_database_with_all_features(conn):
    return pd.read_sql_query(
        "SELECT wine.wine_id, wine.wine_name, wine.type, wine.country, wine.region, wine.alcohol_content, wine.producer, wine.service, wine.volume, wine.grape, wine.harvest, wine.harmonization, wine.image, rating.rating, user.user_id, user.user_name, user.profession, user.gender, user.age  FROM wine INNER JOIN rating ON rating.wine_id = wine.wine_id INNER JOIN user ON user.user_id = rating.user_id;",
        conn)


# Load dataframe with only wines and ratings.
def load_database_with_wines_and_ratings_features(conn):
    return pd.read_sql_query(
        "SELECT wine.wine_id, wine.wine_name, wine.type, wine.country, wine.region, wine.alcohol_content, wine.producer, wine.service, wine.volume, wine.grape, wine.harvest, wine.harmonization, wine.image, rating.rating FROM wine INNER JOIN rating ON rating.wine_id = wine.wine_id;",
        conn)


# Load dataframe with only wines.
def load_database_with_wines_features(conn):
    return pd.read_sql_query("SELECT * FROM wine", conn)


# Load dataframe with only ratings.
def load_database_with_ratings_features(conn):
    return pd.read_sql_query("SELECT * FROM rating", conn)


# This method makes a recommendation with the best wines according to your rating score.
# Returns the best wines with simple recommendation.
@app.route('/v1/recommendations/sample/wines', methods=['GET'])
def get_recommendations_system_simple():
    # Get database connection.
    conn = get_database_connection()

    # Load data frames.
    df = load_database_with_all_features(conn)

    # Close connection.
    close_database_connection(conn)

    # Prepare dataframe
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

    return jsonify(json.loads(result[0:int(10)].to_json(orient='records')))


# This method makes a recommendation with the best wines according to their similarity to other wines.
# wine_key: Name of the wine that was accessed and will be worked on in the recommendation algorithm.
# Returns the best wines with collaborative recommendation.
@app.route('/v1/recommendations/user-collaborative/wines', methods=['GET'])
def get_recommendations_system_collaborative():
    # Key to works.
    try:
        wine_key = request.args.get('wine_key')
    except KeyError:
        return "Wine Key cannot be null"

    # Get database connection
    conn = get_database_connection()

    # Prepare data frames.
    df = load_database_with_wines_and_ratings_features(conn)
    wines = load_database_with_wines_features(conn)
    ratings = load_database_with_ratings_features(conn)

    # Close connection.
    close_database_connection(conn)

    # Average wine ratings
    wines_df_stats = df.groupby('wine_name').agg({'rating': [np.size, np.mean]})

    # Filters the statistical dataset with wines that have more than x analyzes.
    min_10 = wines_df_stats['rating']['size'] >= 3
    wines_df_stats[min_10].sort_values([('rating', 'mean')], ascending=False)

    # Pivot Table
    matrix_df = ratings.pivot_table(index=['wine_id'], columns=['user_id'], values=['rating']).reset_index(drop=True)
    matrix_df.fillna(0, inplace=True)

    # Calculate Cosine Similarity.
    wines_similarity = 1 - pairwise_distances(matrix_df.to_numpy(), metric='cosine')
    np.fill_diagonal(wines_similarity, 0)

    # Set Similarities to Matrix data frame.
    matrix_df = pd.DataFrame(wines_similarity)

    # Recommendation System
    try:
        wine = wines[wines['wine_name'] == wine_key].index.tolist()
        wine = wine[0]

        wines['similarity'] = matrix_df.iloc[wine]
        wines.drop_duplicates('wine_id')
        wines.drop(
            columns=['alcohol_content', 'country', 'grape', 'harmonization', 'harvest', 'image', 'producer', 'region',
                     'service', 'type', 'volume'], axis=1, inplace=True)

        result = pd.DataFrame(wines.sort_values(['similarity'], ascending=False))

        return jsonify(json.loads(result[0:int(10)].to_json(orient='records')))

    except:
        return "Wine not found."

# TThis method recommends other wines based on their content - harmonization.
# wine_key: Name of the wine that was accessed and will be worked on in the recommendation algorithm.
# Returns the best wines with based-content recommendation.
@app.route('/v1/recommendations/based-content/wines', methods=['GET'])
def get_recommendations_system_based_content():
    # Key to works.
    try:
        wine_key = request.args.get('wine_key')
    except KeyError:
        return "Wine Key cannot be null"

    # Get database connection
    conn = get_database_connection()

    # Load data frame.
    wines = load_database_with_wines_features(conn)

    # Close connection.
    close_database_connection(conn)

    # Prepare data frame.
    wines['harmonization'] = wines['harmonization'].fillna('')
    features = ['type', 'country', 'region', 'producer', 'grape']
    for feature in features:
        wines[feature] = wines[feature].apply(clean_data)
    wines['soup'] = wines.apply(create_soup, axis=1)

    # Import CountVectorizer and create the count matrix
    count = CountVectorizer(stop_words=['portuguese', 'english'])
    count_matrix = count.fit_transform(wines['soup'])

    # Compute the Cosine Similarity matrix based on the count_matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    # Reset index of your main DataFrame and construct reverse mapping as before
    metadata = wines.reset_index()
    indices = pd.Series(metadata.index, index=metadata['wine_name'])

    # Get the index of the movie that matches the title
    idx = indices[wine_key]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[0:10]

    # Get the movie indices
    wines_indices = [i[0] for i in sim_scores]

    result = wines.iloc[wines_indices]
    result.drop(columns=['alcohol_content', 'country', 'grape', 'harmonization', 'harvest', 'image', 'producer', 'region',
                     'service', 'type', 'volume', 'soup'], axis=1, inplace=True)

    # Return the top 10 most similar movies
    return jsonify(json.loads(result.to_json(orient='records')))


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 80

    app.run(host='127.0.0.1', port=port, debug=True)
