import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MovieRecommendationSystem:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self.user_movie_matrix = None
        self.svd = None
        
    def load_sample_data(self):
        """Create sample movie data for demonstration"""
        # Sample movies data
        movies_data = {
            'movieId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'title': [
                'The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 
                'Pulp Fiction', 'Forrest Gump', 'Inception', 'The Matrix',
                'Goodfellas', 'The Silence of the Lambs', 'Saving Private Ryan'
            ],
            'genres': [
                'Drama|Crime', 'Crime|Drama', 'Action|Crime|Drama',
                'Crime|Drama', 'Drama|Romance', 'Action|Sci-Fi|Thriller',
                'Action|Sci-Fi', 'Crime|Drama', 'Crime|Drama|Thriller',
                'Drama|War'
            ],
            'description': [
                'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
                'The aging patriarch of an organized crime dynasty transfers control to his reluctant son.',
                'Batman faces the Joker, a criminal mastermind who seeks to undermine Batman and create chaos.',
                'The lives of two mob hitmen, a boxer, and a pair of diner bandits intertwine in four tales of violence.',
                'The presidencies of Kennedy and Johnson, the Vietnam War, and other events shape the life of an Alabama man.',
                'A thief who steals corporate secrets enters dreams to plant an idea in a CEO\'s mind.',
                'A computer hacker learns from mysterious rebels about the true nature of his reality.',
                'The story of Henry Hill and his life in the mob, covering his relationship with his wife and mob partners.',
                'A young FBI cadet must receive the help of an incarcerated cannibal killer to help catch another serial killer.',
                'Following the Normandy Landings, a group of U.S. soldiers go behind enemy lines to retrieve a paratrooper.'
            ]
        }
        
        # Sample ratings data
        ratings_data = {
            'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            'movieId': [1, 2, 3, 2, 4, 5, 1, 3, 6, 4, 7, 8, 5, 9, 10],
            'rating': [5, 4, 3, 5, 4, 3, 4, 5, 4, 3, 5, 4, 4, 5, 3]
        }
        
        self.movies_df = pd.DataFrame(movies_data)
        self.ratings_df = pd.DataFrame(ratings_data)
        
        print("Sample data loaded successfully!")
        print(f"Movies: {len(self.movies_df)}")
        print(f"Ratings: {len(self.ratings_df)}")
        
    def load_from_files(self, movies_file, ratings_file):
        """Load data from CSV files (for real dataset)"""
        try:
            self.movies_df = pd.read_csv(movies_file)
            self.ratings_df = pd.read_csv(ratings_file)
            print("Data loaded from files successfully!")
        except FileNotFoundError:
            print("Files not found. Using sample data instead.")
            self.load_sample_data()
    
    def prepare_content_based_model(self):
        """Prepare the content-based filtering model"""
        if self.movies_df is None:
            self.load_sample_data()
        
        # Combine features for content-based filtering
        self.movies_df['content'] = self.movies_df['genres'] + ' ' + self.movies_df['description']
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['content'])
        
        # Compute cosine similarity matrix
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        
        # Create indices for movie titles
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['title']).drop_duplicates()
        
        print("Content-based model prepared!")
    
    def prepare_collaborative_model(self):
        """Prepare the collaborative filtering model"""
        if self.ratings_df is None:
            self.load_sample_data()
        
        # Create user-movie matrix
        self.user_movie_matrix = self.ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        # Apply SVD for dimensionality reduction
        self.svd = TruncatedSVD(n_components=2)
        self.user_factors = self.svd.fit_transform(self.user_movie_matrix)
        self.movie_factors = self.svd.components_.T
        
        print("Collaborative filtering model prepared!")
    
    def get_content_based_recommendations(self, title, n_recommendations=5):
        """Get recommendations based on movie content similarity"""
        if self.cosine_sim is None:
            self.prepare_content_based_model()
        
        try:
            # Get the index of the movie
            idx = self.indices[title]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            
            # Sort movies based on similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get scores of top n similar movies (skip the first one as it's the input movie itself)
            sim_scores = sim_scores[1:n_recommendations+1]
            
            # Get movie indices
            movie_indices = [i[0] for i in sim_scores]
            
            # Return top n similar movies
            recommendations = self.movies_df[['movieId', 'title', 'genres']].iloc[movie_indices]
            recommendations['similarity_score'] = [i[1] for i in sim_scores]
            
            return recommendations
            
        except KeyError:
            print(f"Movie '{title}' not found in database.")
            return None
    
    def get_collaborative_recommendations(self, user_id, n_recommendations=5):
        """Get recommendations based on collaborative filtering"""
        if self.svd is None:
            self.prepare_collaborative_model()
        
        try:
            # Get user's ratings
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
            
            # Get movies not rated by user
            rated_movies = user_ratings['movieId'].tolist()
            all_movies = self.movies_df['movieId'].tolist()
            unrated_movies = [movie for movie in all_movies if movie not in rated_movies]
            
            if not unrated_movies:
                return pd.DataFrame(columns=['movieId', 'title', 'genres', 'predicted_rating'])
            
            # Predict ratings for unrated movies
            predictions = []
            for movie_id in unrated_movies:
                if movie_id in self.user_movie_matrix.columns:
                    movie_idx = list(self.user_movie_matrix.columns).index(movie_id)
                    user_idx = list(self.user_movie_matrix.index).index(user_id)
                    
                    # Simple prediction using dot product of user and movie factors
                    pred_rating = np.dot(self.user_factors[user_idx], self.movie_factors[movie_idx])
                    predictions.append((movie_id, pred_rating))
            
            # Sort by predicted rating
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Get top recommendations
            top_predictions = predictions[:n_recommendations]
            
            # Create recommendations dataframe
            recommendations = []
            for movie_id, pred_rating in top_predictions:
                movie_info = self.movies_df[self.movies_df['movieId'] == movie_id][['movieId', 'title', 'genres']].iloc[0]
                recommendations.append({
                    'movieId': movie_info['movieId'],
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'predicted_rating': pred_rating
                })
            
            return pd.DataFrame(recommendations)
            
        except KeyError:
            print(f"User ID {user_id} not found in database.")
            return None
    
    def get_hybrid_recommendations(self, user_id=None, title=None, n_recommendations=5):
        """Get hybrid recommendations combining both methods"""
        recommendations = []
        
        # Content-based recommendations
        if title:
            content_recs = self.get_content_based_recommendations(title, n_recommendations)
            if content_recs is not None:
                content_recs['method'] = 'Content-Based'
                recommendations.append(content_recs)
        
        # Collaborative recommendations
        if user_id:
            collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations)
            if collab_recs is not None and not collab_recs.empty:
                collab_recs['method'] = 'Collaborative'
                recommendations.append(collab_recs)
        
        if recommendations:
            return pd.concat(recommendations, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def search_movies(self, query):
        """Search movies by title or genre"""
        if self.movies_df is None:
            self.load_sample_data()
        
        mask = (self.movies_df['title'].str.contains(query, case=False, na=False) | 
                self.movies_df['genres'].str.contains(query, case=False, na=False))
        
        return self.movies_df[mask][['movieId', 'title', 'genres']]
    
    def get_user_history(self, user_id):
        """Get rating history for a user"""
        if self.ratings_df is None:
            self.load_sample_data()
        
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        user_history = user_ratings.merge(self.movies_df, on='movieId')
        
        return user_history[['title', 'genres', 'rating']]

def main():
    """Main function to demonstrate the recommendation system"""
    # Initialize the recommendation system
    recommender = MovieRecommendationSystem()
    
    # Load sample data
    recommender.load_sample_data()
    
    # Prepare models
    recommender.prepare_content_based_model()
    recommender.prepare_collaborative_model()
    
    print("=" * 60)
    print("MOVIE RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Content-based recommendations (by movie title)")
        print("2. Collaborative filtering recommendations (by user ID)")
        print("3. Hybrid recommendations")
        print("4. Search movies")
        print("5. View user rating history")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            print("\nAvailable movies:")
            for idx, row in recommender.movies_df.iterrows():
                print(f"{row['movieId']}. {row['title']} ({row['genres']})")
            
            title = input("\nEnter movie title: ").strip()
            n_recs = int(input("Number of recommendations (default 5): ") or 5)
            
            recommendations = recommender.get_content_based_recommendations(title, n_recs)
            if recommendations is not None:
                print(f"\nRecommendations similar to '{title}':")
                print("-" * 50)
                for idx, row in recommendations.iterrows():
                    print(f"{idx+1}. {row['title']} ({row['genres']}) - Similarity: {row['similarity_score']:.3f}")
        
        elif choice == '2':
            user_id = int(input("Enter user ID: "))
            n_recs = int(input("Number of recommendations (default 5): ") or 5)
            
            recommendations = recommender.get_collaborative_recommendations(user_id, n_recs)
            if recommendations is not None and not recommendations.empty:
                print(f"\nRecommendations for user {user_id}:")
                print("-" * 50)
                for idx, row in recommendations.iterrows():
                    print(f"{idx+1}. {row['title']} ({row['genres']}) - Predicted rating: {row['predicted_rating']:.2f}")
            else:
                print("No recommendations found or user not found.")
        
        elif choice == '3':
            user_id = input("Enter user ID (press Enter to skip): ").strip()
            title = input("Enter movie title (press Enter to skip): ").strip()
            
            if not user_id and not title:
                print("Please provide at least user ID or movie title.")
                continue
            
            user_id = int(user_id) if user_id else None
            n_recs = int(input("Number of recommendations (default 5): ") or 5)
            
            recommendations = recommender.get_hybrid_recommendations(user_id, title, n_recs)
            if not recommendations.empty:
                print(f"\nHybrid Recommendations:")
                print("-" * 50)
                for idx, row in recommendations.iterrows():
                    method = row['method']
                    if method == 'Content-Based':
                        score_info = f"Similarity: {row['similarity_score']:.3f}"
                    else:
                        score_info = f"Predicted rating: {row['predicted_rating']:.2f}"
                    print(f"{idx+1}. {row['title']} ({row['genres']}) - {method} - {score_info}")
            else:
                print("No recommendations found.")
        
        elif choice == '4':
            query = input("Enter search query (title or genre): ").strip()
            results = recommender.search_movies(query)
            if not results.empty:
                print(f"\nSearch results for '{query}':")
                print("-" * 50)
                for idx, row in results.iterrows():
                    print(f"{row['movieId']}. {row['title']} ({row['genres']})")
            else:
                print("No movies found matching your search.")
        
        elif choice == '5':
            user_id = int(input("Enter user ID: "))
            history = recommender.get_user_history(user_id)
            if not history.empty:
                print(f"\nRating history for user {user_id}:")
                print("-" * 50)
                for idx, row in history.iterrows():
                    print(f"{row['title']} - Rating: {row['rating']}/5")
            else:
                print("No rating history found for this user.")
        
        elif choice == '6':
            print("Thank you for using the Movie Recommendation System!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()