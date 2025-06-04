import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords (runs only once)
nltk.download('stopwords')

# Built-in movie dataset (200 movies)
def load_data():
    movies = pd.DataFrame([
        {"title": "The Shawshank Redemption", "genre": "drama", "tags": "prison hope friendship redemption", "year": 1994, "rating": 9.3},
        {"title": "The Godfather", "genre": "crime,drama", "tags": "mafia family power betrayal", "year": 1972, "rating": 9.2},
        {"title": "The Dark Knight", "genre": "action,crime,drama", "tags": "batman joker chaos hero", "year": 2008, "rating": 9.0},
        {"title": "Pulp Fiction", "genre": "crime,drama", "tags": "nonlinear violence gangsters dark comedy", "year": 1994, "rating": 8.9},
        {"title": "Fight Club", "genre": "drama", "tags": "mental illness anarchy soap", "year": 1999, "rating": 8.8},
        {"title": "Inception", "genre": "action,sci-fi", "tags": "dreams heist subconscious", "year": 2010, "rating": 8.8},
        {"title": "The Matrix", "genre": "action,sci-fi", "tags": "simulation reality chosen one", "year": 1999, "rating": 8.7},
        {"title": "Forrest Gump", "genre": "drama,romance", "tags": "simple man historical events love", "year": 1994, "rating": 8.8},
        {"title": "Interstellar", "genre": "adventure,sci-fi", "tags": "space time love black hole", "year": 2014, "rating": 8.6},
        {"title": "The Avengers", "genre": "action,sci-fi", "tags": "superheroes team save world", "year": 2012, "rating": 8.0},
        # Additional 190 movies would be listed here in a real implementation
        # I'm showing 10 for brevity, but you should expand this list
    ])
    
    # Generate more movies for demonstration (in a real app, use actual data)
    genres = ["action", "comedy", "drama", "horror", "sci-fi", "romance", "thriller", "animation"]
    for i in range(190):
        year = 1980 + (i % 40)
        rating = round(5 + (i % 50)/10, 1)
        movies.loc[len(movies)] = {
            "title": f"Sample Movie {i+11}",
            "genre": ",".join([genres[i%8], genres[(i+2)%8]]),
            "tags": "sample tags keywords",
            "year": year,
            "rating": rating
        }
    return movies

# Text preprocessing
def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Recommendation engine
def get_recommendations(movie_title, movies, top_n=5):
    movies['combined_features'] = movies['genre'] + ' ' + movies['tags']
    movies['combined_features'] = movies['combined_features'].apply(preprocess)
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
    
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    idx = indices[movie_title]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]

# Visualization functions
def plot_genre_distribution(movies):
    genre_counts = movies['genre'].str.split(',').explode().value_counts()
    fig = px.bar(genre_counts, 
                 x=genre_counts.index, 
                 y=genre_counts.values,
                 title="🎭 Movie Genre Distribution",
                 labels={'x': 'Genre', 'y': 'Count'})
    st.plotly_chart(fig, use_container_width=True)

def plot_rating_distribution(movies):
    fig = px.histogram(movies, 
                      x="rating",
                      nbins=20,
                      title="⭐ Rating Distribution",
                      labels={'rating': 'IMDB Rating'})
    st.plotly_chart(fig, use_container_width=True)

def plot_year_rating(movies):
    fig = px.scatter(movies,
                    x="year",
                    y="rating",
                    color="genre",
                    title="📅 Movies by Year and Rating",
                    labels={'year': 'Release Year', 'rating': 'Rating'})
    st.plotly_chart(fig, use_container_width=True)

# Streamlit app
def main():
    st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
    st.title("🍿 Movie Recommendation Engine")
    
    # Load data
    movies = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_genre = st.sidebar.selectbox("Filter by Genre", ["All"] + list(movies['genre'].str.split(',').explode().unique()))
    year_range = st.sidebar.slider("Release Year", 1970, 2023, (1990, 2020))
    min_rating = st.sidebar.slider("Minimum Rating", 1.0, 10.0, 7.0)
    
    # Apply filters
    filtered_movies = movies.copy()
    if selected_genre != "All":
        filtered_movies = filtered_movies[filtered_movies['genre'].str.contains(selected_genre)]
    filtered_movies = filtered_movies[
        (filtered_movies['year'] >= year_range[0]) & 
        (filtered_movies['year'] <= year_range[1]) &
        (filtered_movies['rating'] >= min_rating)
    ]
    
    # Visualizations
    st.header("📊 Movie Statistics")
    col1, col2 = st.columns(2)
    with col1:
        plot_genre_distribution(filtered_movies)
    with col2:
        plot_rating_distribution(filtered_movies)
    plot_year_rating(filtered_movies)
    
    # Recommendation system
    st.header("🎯 Get Recommendations")
    selected_movie = st.selectbox("Select a movie you like:", 
                                sorted(filtered_movies['title'].unique()))
    
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(selected_movie, filtered_movies)
        
        st.subheader(f"Because you liked: {selected_movie}")
        st.write("You might enjoy these movies:")
        
        cols = st.columns(5)
        for idx, (_, row) in enumerate(recommendations.iterrows()):
            with cols[idx % 5]:
                st.image("https://via.placeholder.com/150x225.png?text=Movie+Poster", 
                        caption=row['title'], width=150)
                st.caption(f"⭐ {row['rating']} | {row['year']}")
                st.caption(f"🎭 {row['genre'].split(',')[0]}")

if __name__ == "__main__":
    main()