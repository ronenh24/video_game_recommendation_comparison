# Author: Ronen H

from preprocess_video_game_data import preprocess_video_game_data
import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


class VideoGameRecommender:
    def __init__(self, metadata_path: str, reviews_path: str, network_path: str) -> None:
        '''
        Initializes the video game recommender.

        metadata_path: File path to video game metadata.
        reviews_path: File path to video game reviews data.
        network_path: File path to video game network.
        '''
        preprocessed_video_game_data = preprocess_video_game_data(metadata_path, reviews_path, network_path)
        self.game_titles = preprocessed_video_game_data[0]
        self.game_categories = preprocessed_video_game_data[1]
        self.games = preprocessed_video_game_data[2]
        self.game_info_df = preprocessed_video_game_data[3]
        self.users = preprocessed_video_game_data[4]
        self.user_info_df = preprocessed_video_game_data[5]
        video_game_network = nx.from_pandas_edgelist(preprocessed_video_game_data[6], source='from', target='to',
                                                     edge_attr='weight', create_using=nx.DiGraph())
        self.recommendation_sets, self.game_recommendation_set_dict = get_recommendation_sets(video_game_network)
    
    def print_user_info(self, user: str) -> pd.DataFrame:
        '''
        Prints the user review information.
        
        user: User ID.

        Prints the user review information and return as pandas DataFrame.
        '''
        print('User: ' + user)
        
        if user not in self.users:
            print()
            print('Not a user.')
            return pd.DataFrame()
        
        # Is a user.
        user_info = self.users[user]
        user_info_game_titles = []
        user_info_game_categories = []
        for game, rating, vote, verified in zip(self.user_info['game'], self.user_info['rating'],
                                                self.user_info['vote'], self.user_info['verified']): # Obtain information about each review.
            game_title = ''
            if game in self.game_titles:
                game_title = self.game_titles[game]
            user_info_game_titles.append(game_title)
            game_category = set()
            if game in self.game_categories:
                game_category = self.game_categories[game]
            user_info_game_categories.append(game_category)
            print()
            print('Game: ' + game)
            print('Game Title: ' + game_title)
            print('Game Categories: ' + str(game_category))
            print('Rating: ' + str(rating))
            print('Vote: ' + str(vote))
            print('Verified: ' + str(verified))
        return pd.DataFrame({'game': user_info['game'], 'title': user_info_game_titles, 'category': user_info_game_categories,
                             'rating': user_info['rating'], 'vote': user_info['vote'], 'verified': user_info['verified']})
    
    def recommend_by_game_category(self, user: str) -> pd.DataFrame:
        '''
        Obtains the top 10 recommendations for the user based on
        Jaccard similarity of video game categories.

        user: User ID.

        Prints the top 10 recommendations for the user and returns as pandas DataFrame.
        '''
        print('User: ' + user)

        if self.user_info_df[self.user_info_df['user'] == user].empty:
            print()
            print('Not a user.')
            return pd.DataFrame()
        
        # Is a user.
        user_df = self.user_info_df[self.user_info_df['user'] == user]
        user_games = set(user_df['game'].iloc[0])
        user_categories = list(self.game_info_df[self.game_info_df['game'].isin(user_games)]['category'])

        candidate_games = set()
        seen_categories = set()
        for user_category in user_categories:
            if tuple(user_category) not in seen_categories: # Not duplicate set of categories.
                seen_categories.add(tuple(user_category))
                modified_game_df = self.game_info_df[~self.game_info_df['game'].isin(user_games)]
                modified_game_df['number_reviews'] = modified_game_df['user'].apply(len)
                modified_game_df['average_rating'] = modified_game_df['rating'].apply(np.mean)
                modified_game_df['jaccard_similarity'] = modified_game_df['category'].apply(
                    lambda other_category: jaccard_similarity(user_category, other_category))
                # Top 10 by Jaccard similarity then average rating then popularity to add to set of candidates.
                top_games = list(modified_game_df[modified_game_df['jaccard_similarity'] >= 0.5].sort_values(
                    ['jaccard_similarity', 'average_rating', 'number_reviews'], ascending=False).drop_duplicates('title')['game'].iloc[:10])
                for top_game in top_games:
                    candidate_games.add(top_game)
        
        candidate_game_df = self.game_info_df[self.game_info_df['game'].isin(candidate_games)]
        candidate_game_df['number_reviews'] = candidate_game_df['user'].apply(len)
        candidate_game_df['average_rating'] = candidate_game_df['rating'].apply(np.mean)
        # Top 10 from candidates by popularity then average rating to recommend.
        recommended_games = candidate_game_df.sort_values(['number_reviews', 'average_rating'], ascending=False).drop(
            columns=['user', 'rating', 'vote', 'verified', 'average_rating', 'number_reviews']).iloc[:10]
        for recommended_game, recommended_game_title, recommended_game_categories in recommended_games.itertuples(index=False, name=None):
                print()
                print('Game: ' + recommended_game)
                print('Game Title: ' + recommended_game_title)
                print('Game Categories: ' + str(recommended_game_categories))
        return recommended_games
    
    def recommend_by_rating_influence_reliability(self, user: str) -> pd.DataFrame:
        '''
        Obtains the top 10 recommendations for the user based on recommendation sets
        from Louvain community detection on video game network.

        user: User ID.

        Returns the top 10 recommendations for the user.
        '''
        print('User: ' + user)

        if self.user_info_df[self.user_info_df['user'] == user].empty:
            print()
            print('Not a user.')
            return pd.DataFrame()
        
        # Is a user.
        user_df = self.user_info_df[self.user_info_df['user'] == user]
        user_games = set(user_df['game'].iloc[0])

        candidate_games = set()
        for user_game in user_games:
            if user_game in self.game_recommendation_set_dict: # Video game in network.
                recommendation_set = self.recommendation_sets[self.game_recommendation_set_dict[user_game]]
                modified_game_df = self.game_info_df[~self.game_info_df['game'].isin(user_games)]
                modified_game_df = modified_game_df[modified_game_df['game'].isin(recommendation_set)]
                modified_game_df['number_reviews'] = modified_game_df['user'].apply(len)
                modified_game_df['average_rating'] = modified_game_df['rating'].apply(np.mean)
                # Top 10 in recommendation set by average rating then popularity to add to set of candidates.
                top_games = list(modified_game_df.sort_values(
                    ['average_rating', 'number_reviews'], ascending=False).drop_duplicates('title')['game'].iloc[:10])
                for top_game in top_games:
                    candidate_games.add(top_game)
        
        if len(candidate_games) == 0:
            print()
            print('Using recommender by similar video game categories.')
            print()
            return self.recommend_by_game_category(user)
        
        # At least one candidate game.
        candidate_game_df = self.game_info_df[self.game_info_df['game'].isin(candidate_games)]
        candidate_game_df['number_reviews'] = candidate_game_df['user'].apply(len)
        candidate_game_df['average_rating'] = candidate_game_df['rating'].apply(np.mean)
        # Top 10 from candidates by popularity then average rating to recommend.
        recommended_games = candidate_game_df.sort_values(['number_reviews', 'average_rating'], ascending=False).drop(
            columns=['user', 'rating', 'vote', 'verified', 'average_rating', 'number_reviews']).iloc[:10]
        for recommended_game, recommended_game_title, recommended_game_categories in recommended_games.itertuples(index=False, name=None):
                print()
                print('Game: ' + recommended_game)
                print('Game Title: ' + recommended_game_title)
                print('Game Categories: ' + str(recommended_game_categories))
        return recommended_games
    
    def recommend_by_user_ratings(self, user: str) -> pd.DataFrame:
        '''
        Obtains the top 10 recommendations for the user based on liked video games
        from similar user ratings by Pearson correlation coefficient.

        user: User ID.

        Returns the top 10 recommendations for the user.
        '''
        print('User: ' + user)

        if self.user_info_df[self.user_info_df['user'] == user].empty:
            print()
            print('Not a user.')
            return pd.DataFrame()
        
        # Is a user.
        user_df = self.user_info_df[self.user_info_df['user'] == user]
        user_games = set(user_df['game'].iloc[0])

        if len(user_games) == 1:
            print()
            print('Using recommender by similar video game categories.')
            print()
            return self.recommend_by_game_category(user)
        
        # Reviewed at least two different games.
        other_users_df = self.user_info_df[self.user_info_df['user'] != user]
        other_users_df['pearson_coefficient'] = other_users_df[['game', 'rating']].apply(
            lambda curr_user: pearson_coefficient(user_df['game'].iloc[0], user_df['rating'].iloc[0],
                                                  curr_user['game'], curr_user['rating']), axis=1)
        other_users_df['number_unique'] = other_users_df['game'].apply(lambda game: len(set(game)) - len(set(game).intersection(user_games)))
        other_users_df['number_reviews'] = other_users_df['game'].apply(len)
        # Most similar users in terms of rating by Pearson Correlation Coefficient.
        similar_users = list(other_users_df[
            (other_users_df['number_unique'] > 0) & (other_users_df['pearson_coefficient'] >= 0.7)].sort_values(
                ['pearson_coefficient', 'number_reviews'], ascending=False)['user'])
        
        candidate_games = set()
        for similar_user in similar_users:
            similar_user_df = other_users_df[other_users_df['user'] == similar_user]
            for similar_user_game, similar_user_rating in zip(similar_user_df['game'].iloc[0], similar_user_df['rating'].iloc[0]):
                if similar_user_game not in user_games and similar_user_rating >= 4:
                    candidate_games.add(similar_user_game) # Add similar user liked game to candidates.
        
        if len(candidate_games) == 0:
            print()
            print('Using recommender by similar video game categories.')
            print()
            return self.recommend_by_game_category(user)
        
        # At least one candidate game.
        candidate_game_df = self.game_info_df[self.game_info_df['game'].isin(candidate_games)]
        candidate_game_df['number_reviews'] = candidate_game_df['user'].apply(len)
        candidate_game_df['average_rating'] = candidate_game_df['rating'].apply(np.mean)
        # Top 10 from candidates by popularity then average rating to recommend.
        recommended_games = candidate_game_df.sort_values(['number_reviews', 'average_rating'], ascending=False).drop(
            columns=['user', 'rating', 'vote', 'verified', 'number_reviews', 'average_rating']).iloc[:10]
        for recommended_game, recommended_game_title, recommended_game_categories in recommended_games.itertuples(index=False, name=None):
                print()
                print('Game: ' + recommended_game)
                print('Game Title: ' + recommended_game_title)
                print('Game Categories: ' + str(recommended_game_categories))
        return recommended_games


def jaccard_similarity(first_categories: set[str], second_categories: set[str]) -> float:
        '''
        Calculates the Jaccard Similarity between the first video game categories
        and second video game categories.

        first_categories: First video game categories.
        second_categories: Second video game categories.

        Returns the Jaccard Similarity.
        '''
        num_shared = len(first_categories.intersection(second_categories))
        num_total = len(first_categories.union(second_categories))

        if num_total == 0: # Both video games have no categories.
            return 1
        
        return num_shared / num_total

def get_recommendation_sets(video_games_network: nx.DiGraph, resolution: float=2.5) -> tuple[list[set[str]], dict[str, int]]:
    '''
    Obtain video game recommendation sets as determined by Louvain community detection on the video game network.

    video_games_network: Video game network of weighted average rating.

    Return video game recommendation sets.
    '''
    recommendation_sets = nx.community.louvain_communities(video_games_network, resolution=resolution, seed=100)
    game_recommendation_set_dict = {}
    for i, recommendation_set in enumerate(recommendation_sets):
        for recommendation_set_game in recommendation_set:
            game_recommendation_set_dict[recommendation_set_game] = i
    return recommendation_sets, game_recommendation_set_dict

def pearson_coefficient(first_user_games: list[str], first_user_ratings: list[int],
                        second_user_games: list[str], second_user_ratings: list[int]) -> float:
    '''
    Calculates the Pearson correlation coefficient between the first user and second user in terms of their ratings.

    first_user_games: First user games.
    first_user_ratings: First user ratings for games.
    second_user_games: Second user games.
    second_user_ratings: Second user ratings for games.

    Returns the Pearson correlation coefficient.
    '''
    first_user_game_rating = {}
    for first_user_game, first_user_rating in zip(first_user_games, first_user_ratings):
        if first_user_game not in first_user_game_rating:
            first_user_game_rating[first_user_game] = first_user_rating
        else:
            first_user_game_rating[first_user_game] = max(first_user_game_rating[first_user_game], first_user_rating)
    second_user_game_rating = {}
    for second_user_game, second_user_rating in zip(second_user_games, second_user_ratings):
        if second_user_game not in second_user_game_rating:
            second_user_game_rating[second_user_game] = second_user_rating
        else:
            second_user_game_rating[second_user_game] = max(second_user_game_rating[second_user_game], second_user_rating)

    shared_games = set(first_user_game_rating.keys()).intersection(set(second_user_game_rating.keys()))
    if len(shared_games) < 2:
        return 0

    # At least two video games in common.
    first_user_ratings_shared = []
    second_user_ratings_shared = []
    for shared_game in shared_games:
        first_user_ratings_shared.append(first_user_game_rating[shared_game])
        second_user_ratings_shared.append(second_user_game_rating[shared_game])
    
    return pearsonr(first_user_ratings_shared, second_user_ratings_shared).statistic

