import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Page configuration
st.set_page_config(page_title="Billboard Music Analysis", layout="wide")

# Title
st.title("🎵 2010-2019 Popular Songs Analysis")
st.write("Explore the characteristics and trends of popular songs over the past decade")

# Create complete sample dataset
@st.cache_data
def load_data():
    # Sample data based on real Billboard data characteristics
    np.random.seed(42)
    years = list(range(2010, 2020))
    
    data = []
    for year in years:
        for i in range(60):  # 60 popular songs per year
            song_id = f"{year}_{i+1}"
            
            # Generate realistic music data
            bpm = np.random.randint(80, 180)
            energy = np.random.randint(50, 100)
            danceability = np.random.randint(40, 95)
            popularity = np.random.randint(70, 100)
            
            # Simulate music feature trends (changes over time)
            if year >= 2015:
                bpm += 10  # Faster tempo in recent years
                energy += 5
            
            # Artists and genres
            artists = ['Ed Sheeran', 'Taylor Swift', 'Drake', 'Ariana Grande', 'The Weeknd', 
                      'Billie Eilish', 'Post Malone', 'Dua Lipa', 'Bruno Mars', 'Rihanna']
            genres = ['Pop', 'Hip-Hop', 'R&B', 'Electronic', 'Rock', 'Country']
            
            data.append({
                'id': song_id,
                'title': f"Song_{song_id}",
                'artist': np.random.choice(artists),
                'year': year,
                'bpm': max(60, min(200, bpm + np.random.randint(-10, 10))),
                'energy': max(30, min(100, energy + np.random.randint(-10, 10))),
                'danceability': max(20, min(99, danceability + np.random.randint(-10, 10))),
                'popularity': popularity,
                'genre': np.random.choice(genres),
                'duration_ms': np.random.randint(180000, 240000)  # 3-4 minutes
            })
    
    return pd.DataFrame(data)

df = load_data()

# Sidebar - Data filtering
st.sidebar.header("🎛️ Data Filtering")

# Year selection
selected_years = st.sidebar.slider(
    "Select Year Range",
    min_value=int(df['year'].min()),
    max_value=int(df['year'].max()),
    value=(2015, 2019)
)

# Genre selection
all_genres = ['All'] + sorted(df['genre'].unique().tolist())
selected_genre = st.sidebar.selectbox("Select Music Genre", all_genres)

# Apply filters
filtered_df = df[
    (df['year'] >= selected_years[0]) & 
    (df['year'] <= selected_years[1])
]

if selected_genre != 'All':
    filtered_df = filtered_df[filtered_df['genre'] == selected_genre]

# Main display area
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Songs", len(filtered_df))
with col2:
    st.metric("Year Range", f"{filtered_df['year'].min()}-{filtered_df['year'].max()}")
with col3:
    st.metric("Number of Artists", filtered_df['artist'].nunique())
with col4:
    st.metric("Number of Genres", filtered_df['genre'].nunique())

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Trend Analysis", "🎵 Music Features", "🏆 Leaderboard"])

with tab1:
    st.subheader("Data Preview")
    st.dataframe(filtered_df.head(10))
    
    # Basic statistics
    st.subheader("Data Statistics")
    st.write(filtered_df[['bpm', 'energy', 'danceability', 'popularity']].describe())

with tab2:
    st.subheader("Music Feature Trends by Year")
    
    # Calculate averages by year
    yearly_avg = filtered_df.groupby('year').agg({
        'bpm': 'mean',
        'energy': 'mean', 
        'danceability': 'mean',
        'popularity': 'mean'
    }).reset_index()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # BPM trend
    ax1.plot(yearly_avg['year'], yearly_avg['bpm'], marker='o', linewidth=2, markersize=8)
    ax1.set_title('Average BPM Trend')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('BPM')
    ax1.grid(True, alpha=0.3)
    
    # Energy trend
    ax2.plot(yearly_avg['year'], yearly_avg['energy'], marker='s', color='orange', linewidth=2, markersize=8)
    ax2.set_title('Average Energy Trend')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Energy')
    ax2.grid(True, alpha=0.3)
    
    # Danceability trend
    ax3.plot(yearly_avg['year'], yearly_avg['danceability'], marker='^', color='green', linewidth=2, markersize=8)
    ax3.set_title('Average Danceability Trend')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Danceability')
    ax3.grid(True, alpha=0.3)
    
    # Popularity trend
    ax4.plot(yearly_avg['year'], yearly_avg['popularity'], marker='d', color='red', linewidth=2, markersize=8)
    ax4.set_title('Average Popularity Trend')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Popularity')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("Music Feature Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature = st.selectbox("Select Feature", ['bpm', 'energy', 'danceability', 'popularity'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(filtered_df[feature], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel(feature)
        ax.set_ylabel('Number of Songs')
        ax.set_title(f'{feature} Distribution')
        st.pyplot(fig)
    
    with col2:
        # Genre distribution
        genre_counts = filtered_df['genre'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('Music Genre Distribution')
        st.pyplot(fig)

with tab4:
    st.subheader("Popular Artists Leaderboard")
    
    # Statistics by artist
    artist_stats = filtered_df.groupby('artist').agg({
        'title': 'count',
        'popularity': 'mean',
        'bpm': 'mean'
    }).round(1).reset_index()
    artist_stats.columns = ['Artist', 'Number of Songs', 'Average Popularity', 'Average BPM']
    artist_stats = artist_stats.sort_values('Number of Songs', ascending=False)
    
    st.dataframe(artist_stats.head(10))
