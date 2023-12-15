# Author
Ronen H

# Data Source
The video game reviews, 'Video_Games.json.gz', and metadata, 'meta_Video_Games.json.gz', can be accessed via [https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/).  

# Video Game Recommenders Notebook
The code to explore the three video game recommenders is in the `exploring_recommenders.ipynb` notebook.  

# User Information
The information of the reviews (game, category, rating, vote, verified) for each user are saved in the **User_Information** directory as CSV files.  

# Results
The top 10 recommendations for each user from the video game recommender based on similar categories are saved in the **Results/recommender_1** directory as CSV files.  

The top 10 recommendations for each user from the video game recommender based on rating, influence, and reliability of reviews are saved in the **Results/recommender_2** directory as CSV files.  

The top 10 recommendations for each user from the video game recommender based on similar user ratings are saved in the **Results/recommender_3** directory as CSV files.  

The bar plot of the number of "good" recommendations is saved in the **Results** directory as **number_good.jpg**.

# Video Game Recommender System
The video game recommender system is represented by the `VideoGameRecommender` class in `video_game_recommender.py`.  

To use, do
```
from video_game_recommender import VideoGameRecommender
video_game_recommender = VideoGameRecommender('meta_Video_Games.json.gz', 'Video_Games.json.gz', 'video_game_network.csv')
```
.  

For a user
```
user_id = '...'
```
- To recommend by similar category, do
    ```
    video_game_recommender.recommend_by_game_category(user_id)
    ```
- To recommend by rating, influence, and reliability of reviews, do
    ```
   video_game_recommender.recommend_by_rating_influence_reliability(user_id)
    ```
- To recommender by similar user ratings, do
    ```
    video_game_recommender.recommend_by_user_ratings(user_id)
    ```
.

# References
Dugué, N., & Perez, A. (2022). Direction matters in complex networks: A theoretical and applied 
study for greedy modularity optimization. Retrieved from 
[https://www.sciencedirect.com/science/article/pii/S0378437122005234](https://www.sciencedirect.com/science/article/pii/S0378437122005234)  
Ni, J., Li, J., & McAuley, J. (2019). Empirical Methods in Natural Language Processing 
(EMNLP). Retrieved from [https://cseweb.ucsd.edu//~jmcauley/pdfs/emnlp19a.pdf](https://cseweb.ucsd.edu//~jmcauley/pdfs/emnlp19a.pdf)
