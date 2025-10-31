# Movie-Recommendation-System
A Python movie recommendation system using machine learning. It suggests films through content-based filtering (finding similar movies by genres/descriptions) and collaborative filtering (analyzing user rating patterns). Built with Pandas and Scikit-learn, it combines both methods for accurate, personalized movie suggestions interactive interface.


# Movie Recommendation System

A comprehensive movie recommendation system built with Python that implements multiple recommendation algorithms to suggest films based on user preferences and movie similarity.

## Features

- **Content-Based Filtering**: Recommends similar movies based on genres and descriptions using TF-IDF and cosine similarity
- **Collaborative Filtering**: Suggests movies using user rating patterns with Singular Value Decomposition (SVD)
- **Hybrid Approach**: Combines both methods for enhanced recommendations
- **Movie Search**: Find movies by title or genre
- **User History**: View rating history for specific users
- **Interactive CLI**: User-friendly command-line interface

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
```

2. Install required packages:
```bash
pip install pandas scikit-learn numpy
```

## Usage

Run the main script:
```bash
python movie_recommender.py
```

Follow the interactive menu to:
- Get content-based recommendations by movie title
- Get collaborative recommendations by user ID
- Use hybrid recommendations combining both methods
- Search movies by title or genre
- View user rating history

## Project Structure

- `movie_recommender.py` - Main implementation file
- Sample dataset included for demonstration
- Modular design for easy extension

## Future Enhancements

- Web interface using Streamlit
- Integration with real movie databases
- Advanced deep learning models
- Real-time recommendation API
