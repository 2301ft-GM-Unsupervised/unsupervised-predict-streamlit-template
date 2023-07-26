<<<<<<< Updated upstream
### Content-based systems
These systems make recommendations using a user's movie and profile features. They hypothesize that if a user was interested in a movie in the past, they will once again be interested in it in the future. Similar items are usually grouped based on their features. User profiles are constructed using historical interactions or by explicitly asking users about their interests.
For instance, in a content-based movie recommender system, the similarity between the movies is calculated based on genres, the actors in the movie, the director of the movie, etc. 

### Collaborative filtering systems
Collaborative filtering is currently one of the most frequently used approaches and usually provides better results than content-based recommendations.
These kinds of systems utilize user interactions to filter for movies of interest. Collaborative filtering systems are based on the assumption that if a user likes movie A and another user likes the same movie A as well as another movie, movie B, the first user could also be interested in the second movie. Hence, they aim to predict new interactions based on historical ones.

This app tries to mimic an everyday simple movie recommender system that uses both algorithms to make personalised movie recommendations for various viewers to achieve an enhanced customer experience.
=======
# Types of Recommender Systems


## Content-based systems

In contrast to collaborative filtering, content-based techniques employ user and item feature vectors to make recommendations. The fundamental differences between the two approaches are that content-based systems recommend items based on content features (no need for data about other users; recommendations about niche items, etc.) whereas collaborative filtering is based on user behaviour only and recommends items based on users with similar patterns (no domain knowledge; serendipity, etc.). A content-based filtering method works by making movie proposals to the user based on the content in the movies. It recognizes that clustering in the collaborative filtering recommendations may not match the preferences of the users. The tastes and preferences of people with similar demographic characteristics are very different; what person X likes may not be similar to what person Y likes to watch. To solve this problem, content-based filtering algorithms give recommendations based on the contents of the movies. In movie recommendations, some of the contents are the key characters and the genre of the movie.

## Collaborative filtering systems

Collaborative filtering works by matching the similarities in items and users. It looks at the characteristics of the users and the characteristics of the items the users have watched or searched for before. In general, latent features obtained from rating matrices are looked at. In movie recommender systems, the recommendations are made based on the user information and what other people with similar user information are watching. For example, collaborative filtering in movie recommender systems picks the user demographic characteristics such as age, gender, and ethnicity. Through these features, movie recommendations are made that match other people with similar demographic characteristics and previous user search history. Collaborative filtering suffers from a cold start if the user has not input any information, or the information is too little for any accurate clustering. In these cases, it does not know what to suggest. The accuracy of the suggestion is also limited because people with similar demographic characteristics may not have similar preferences.

## Context-Based Filtering

This filtering technology is an improvement of the collaborative filtering method. It assumes that if person A and person B hold the same opinion on issue X, it is most likely that the same people will hold the same opinion/preference/thinking on a different issue Z. For example, if both people are attracted to Christmas movies from Netflix, it is most likely that they will still like Christmas movies by Showmax. The context-based filtering method recommends items with similar features or characteristics because the applications have just been extended to a different context. It makes the same suggestions though the contexts are different. In most cases, web browsers import bookmarks and other settings when one upgrades from one browser to the next. This represents a change in context, since most of the settings and other items are imported into the new context, and the data available are used in making useful suggestions. Similarly, movie recommender systems may make a similar recommendation based on data from the previous context. It is worth mentioning here about context-aware recommender systems (CARS), where the concept of context is well defined. CARS acclimatize to the exact condition in which the recommended item will be used. In this respect, CARS could avoid recommending a very long film to a user after a stressful day at work or suggest a romantic film if he/she is in the company of his/her partner.

## Hybrid Filtering

This is a filtering technique that applies the concepts of all the other algorithms. It combines both collaborative filtering, content-based filtering, and context-based filtering to overcome the challenges of each method. It is superior because it achieves higher performance in making the suggestions and also a faster computational time. For instance, collaborative filtering may lack information about domain dependencies while content-based filtering lacks information about the preferences of the people. A combination of these overcomes these challenges since user behaviour data and the content data are used to come up with recommendations.
>>>>>>> Stashed changes

![gif](https://s3.amazonaws.com/codecademy-content/programs/code-foundations-path/ds-survey/utilitymatrix.gif)