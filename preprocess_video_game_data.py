# Author: Ronen H

import os
import gzip
from tqdm import tqdm
import orjson
from collections import Counter
import html
import pandas as pd
import numpy as np


def preprocess_video_game_data(metadata_path: str, reviews_path: str, network_path: str) -> tuple:
    '''
    Preprocesses the video game metadata and reviews data.

    metadata_path: File path to video game metadata.
    reviews_path: File path to video game reviews data.
    network_path: File path to video game network.

    Returns video game information, user information, and video game network.
    '''
    game_titles, game_categories = _obtain_titles_and_common_categories(metadata_path)
    games, game_info_df = _obtain_video_game_info(reviews_path, game_titles, game_categories)
    users, user_info_df = _obtain_user_info(reviews_path)
    video_game_network_df = _obtain_video_game_network(network_path, users)
    return game_titles, game_categories, games, game_info_df, users, user_info_df, video_game_network_df

def _obtain_titles_and_common_categories(metadata_path: str) -> tuple[dict[str, str], dict[str, set[str]]]:
    '''
    Gets the title and common categories for each video game.

    metadata_path: File path to video game metadata.

    Returns the title and common categories for each video game.
    '''
    unique_games = set() # Seen games.
    game_titles = {} # Titles of video games.
    game_category_counts = Counter() # Video game category counts.
    with gzip.open('meta_Video_Games.json.gz') as games_file:
        for json_line in tqdm(games_file):
            game = orjson.loads(json_line)
            game_id = game['asin']
            if game_id not in unique_games: # Not duplicate game.
                unique_games.add(game_id)
                game_titles[game_id] = html.unescape(game['title']).strip()
                categories = set()
                for category in game['category']:
                    category = html.unescape(category).strip()
                    if category == '</span></span></span>': # Anything after most insignificant.
                        break
                    if category != 'Video Games': # Every video game has category 'Video Games'.
                        categories.add(category)
                game_category_counts.update(Counter(categories))

    game_categories = {} # Common categories of video games.
    with gzip.open('meta_Video_Games.json.gz') as games_file:
        for json_line in tqdm(games_file):
            game = orjson.loads(json_line)
            game_id = game['asin']
            if game_id not in game_categories: # Not duplicate game.
                game_categories[game_id] = set()
                for category in game['category']:
                    category = html.unescape(category).strip()
                    # Common if appears at least in 10 video games.
                    if category != 'Video Games' and category != '</span></span></span>' and game_category_counts[category] >= 10:
                        game_categories[game_id].add(category)
    return game_titles, game_categories

def _obtain_video_game_info(reviews_path: str, game_titles: dict[str, str],
                            game_categories: dict[str, set[str]]) -> tuple[dict[str, dict], pd.DataFrame]:
    '''
    Gets the video game information.

    reviews_path: File path to video game reviews data.
    game_titles: Title for each video game.
    game_categories: Common categories for each video game.

    Returns the video game information as dictionary and pandas DataFrame.
    '''
    unique_reviews = set() # Seen reviews.
    games = {} # Information about video games.
    with gzip.open(reviews_path) as reviews_file:
        for json_line in tqdm(reviews_file):
            review = orjson.loads(json_line)
            user = review['reviewerID']
            game = review['asin']
            rating = review['overall']
            vote = 0
            if 'vote' in review:
                vote = int(review['vote'].replace(',', ''))
            verified = review['verified']
            if (user, game, rating, vote, verified) not in unique_reviews: # Not duplicate review.
                unique_reviews.add((user, game, rating, vote, verified))
                if game not in games:
                    games[game] = {'user': [], 'rating': [], 'vote': [], 'verified': []}
                games[game]['user'].append(user)
                games[game]['rating'].append(rating)
                games[game]['vote'].append(vote)
                games[game]['verified'].append(verified)

    games_info = {'game': [], 'title': [], 'user': [], 'rating': [], 'vote': [], 'verified': [], 'category': []}
    for game, game_info in games.items():
        games_info['game'].append(game)
        if game in game_titles:
            games_info['title'].append(game_titles[game])
        else:
            games_info['title'].append('')
        games_info['user'].append(game_info['user'])
        games_info['rating'].append(game_info['rating'])
        games_info['vote'].append(game_info['vote'])
        games_info['verified'].append(game_info['verified'])
        if game in game_categories:
            games_info['category'].append(game_categories[game])
        else:
            games_info['category'].append(set())

    game_info_df = pd.DataFrame(games_info)
    return games, game_info_df


def _obtain_user_info(reviews_path: str) -> dict[dict[str, dict], pd.DataFrame]:
    '''
    Gets the user information.

    reviews_path: File path to video game reviews data.

    Returns the user information as dictionary and pandas DataFrame.
    '''
    unique_reviews = set() # Seen reviews.
    users = {} # Information about users.
    with gzip.open(reviews_path) as reviews_file:
        for json_line in tqdm(reviews_file):
            review = orjson.loads(json_line)
            user = review['reviewerID']
            game = review['asin']
            rating = review['overall']
            vote = 0
            if 'vote' in review:
                vote = int(review['vote'].replace(',', ''))
            verified = review['verified']
            if (user, game, rating, vote, verified) not in unique_reviews: # Not duplicate review.
                unique_reviews.add((user, game, rating, vote, verified))
                if user not in users:
                    users[user] = {'game': [], 'rating': [], 'vote': [], 'verified': []}
                users[user]['game'].append(game)
                users[user]['rating'].append(rating)
                users[user]['vote'].append(vote)
                users[user]['verified'].append(verified)

    users_info = {'user': [], 'game': [], 'rating': [], 'vote': [], 'verified': []}
    for user, user_info in users.items():
        users_info['user'].append(user)
        users_info['game'].append(user_info['game'])
        users_info['rating'].append(user_info['rating'])
        users_info['vote'].append(user_info['vote'])
        users_info['verified'].append(user_info['verified'])

    user_info_df = pd.DataFrame(users_info)
    return users, user_info_df

def _obtain_video_game_network(network_path: str, users: dict[str, dict]) -> pd.DataFrame:
    '''
    Gets the video game network of weighted average rating.

    network_path: File path to video game network.
    users: User information.

    Returns video game network as pandas DataFrame.
    '''
    if not os.path.isfile(network_path):
        edges = {}
        for user, user_info in tqdm(users.items()):
            if len(set(user_info['game'])) >= 2: # User has reviewed at least two different games.
                for i in range(len(user_info['game'])):
                    curr_prod = user_info['game'][i]
                    for j in range(len(user_info['game'])):
                        if i != j:
                            next_prod = user_info['game'][j]
                            if curr_prod != next_prod:
                                if (curr_prod, next_prod) not in edges:
                                    edges[(curr_prod, next_prod)] = []
                                rating = [user_info['rating'][j]]
                                if user_info['vote'][j] > 0: # Count rating by number of votes if any.
                                    rating.extend(rating * user_info['vote'][j])
                                if user_info['verified'][j]: # Count rating twice if verified.
                                    rating.extend(rating)
                                edges[(curr_prod, next_prod)].extend(rating)
        curr_next_weight = {'from': [], 'to': [], 'weight': []}
        for games, rating in tqdm(edges.items()):
            curr_next_weight['from'].append(games[0])
            curr_next_weight['to'].append(games[1])
            curr_next_weight['weight'].append(np.mean(rating))
        pd.DataFrame(curr_next_weight).to_csv(network_path, index=None)

    video_game_network_df = pd.read_csv(network_path)
    return video_game_network_df

