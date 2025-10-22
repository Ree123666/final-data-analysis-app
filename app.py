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
        # Vary the number of songs per year to make it more realistic
        songs_per_year = np.random.randint(55, 65)
        for i in range(songs_per_year):
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
            
            # Artists with different popularity weights
            artists = [
                ('Ed Sheeran', 0.15),      # More popular
                ('Taylor Swift', 0.14),    # More popular
                ('Drake', 0.13),           # More popular
                ('Ariana Grande', 0.12),   # More popular
                ('The Weeknd', 0.11),      # More popular
                ('Billie Eilish', 0.08),   # Less popular (newer artist)
                ('Post Malone', 0.09),     # Medium popularity
                ('Dua Lipa', 0.07),        # Less popular
                ('Bruno Mars', 0.06),      # Less popular in later years
                ('Rihanna', 0.05)          # Less active in later years
            ]
            
            genres = ['Pop', 'Hip-Hop', 'R&B', 'Electronic', 'Rock', 'Country']
            
            # Select artist based on weighted probabilities
            artist_names, weights = zip(*artists)
            selected_artist = np.random.choice(artist_names, p=weights)
            
            data.append({
                'id': song_id,
                'title': f"Song_{song_id}",
                'artist': selected_artist,
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

# Artist selection (optional)
all_artists = ['All'] + sorted(df['artist'].unique().tolist())
selected_artist = st.sidebar.selectbox("Select Artist (Optional)", all_artists)

# Apply filters
filtered_df = df[
    (df['year'] >= selected_years[0]) & 
    (df['year'] <= selected_years[1])
]

if selected_genre != 'All':
    filtered_df = filtered_df[filtered_df['genre'] == selected_genre]

if selected_artist != 'All':
    filtered_df = filtered_df[filtered_df['artist'] == selected_artist]

# Main display area
st.subheader("📊 Dataset Overview")
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
    
    # Data information
    st.subheader("Dataset Information")
    st.write(f"**Total records:** {len(filtered_df)}")
    st.write(f"**Date range:** {filtered_df['year'].min()} - {filtered_df['year'].max()}")
    st.write(f"**Features available:** {', '.join(filtered_df.columns.tolist())}")

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
    ax1.plot(yearly_avg['year'], yearly_avg['bpm'], marker='o', linewidth=2, markersize=8, color='#1f77b4')
    ax1.set_title('Average BPM Trend', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('BPM')
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(yearly_avg['year'], yearly_avg['bpm'], alpha=0.3, color='#1f77b4')
    
    # Energy trend
    ax2.plot(yearly_avg['year'], yearly_avg['energy'], marker='s', color='#ff7f0e', linewidth=2, markersize=8)
    ax2.set_title('Average Energy Trend', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Energy')
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(yearly_avg['year'], yearly_avg['energy'], alpha=0.3, color='#ff7f0e')
    
    # Danceability trend
    ax3.plot(yearly_avg['year'], yearly_avg['danceability'], marker='^', color='#2ca02c', linewidth=2, markersize=8)
    ax3.set_title('Average Danceability Trend', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Danceability')
    ax3.grid(True, alpha=0.3)
    ax3.fill_between(yearly_avg['year'], yearly_avg['danceability'], alpha=0.3, color='#2ca02c')
    
    # Popularity trend
    ax4.plot(yearly_avg['year'], yearly_avg['popularity'], marker='d', color='#d62728', linewidth=2, markersize=8)
    ax4.set_title('Average Popularity Trend', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Popularity')
    ax4.grid(True, alpha=0.3)
    ax4.fill_between(yearly_avg['year'], yearly_avg['popularity'], alpha=0.3, color='#d62728')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Trend insights
    st.subheader("Trend Insights")
    if len(yearly_avg) > 1:
        bpm_change = yearly_avg['bpm'].iloc[-1] - yearly_avg['bpm'].iloc[0]
        energy_change = yearly_avg['energy'].iloc[-1] - yearly_avg['energy'].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("BPM Change", f"{bpm_change:+.1f}")
        with col2:
            st.metric("Energy Change", f"{energy_change:+.1f}")

with tab3:
    st.subheader("Music Feature Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature = st.selectbox("Select Feature", ['bpm', 'energy', 'danceability', 'popularity'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(filtered_df[feature], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel(feature.title())
        ax.set_ylabel('Number of Songs')
        ax.set_title(f'{feature.title()} Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Feature statistics
        st.write(f"**{feature.title()} Statistics:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean", f"{filtered_df[feature].mean():.1f}")
        with col2:
            st.metric("Median", f"{filtered_df[feature].median():.1f}")
        with col3:
            st.metric("Std Dev", f"{filtered_df[feature].std():.1f}")
    
    with col2:
        # Genre distribution
        genre_counts = filtered_df['genre'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(genre_counts)))
        wedges, texts, autotexts = ax.pie(genre_counts.values, labels=genre_counts.index, 
                                         autopct='%1.1f%%', startangle=90, colors=colors)
        ax.set_title('Music Genre Distribution', fontweight='bold')
        
        # Improve readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        st.pyplot(fig)
        
        # Genre statistics
        st.write("**Genre Statistics:**")
        for genre, count in genre_counts.items():
            st.write(f"- {genre}: {count} songs ({count/len(filtered_df)*100:.1f}%)")

with tab4:
    st.subheader("Popular Artists Leaderboard")
    
    # Statistics by artist - fixed counting logic
    artist_stats = filtered_df.groupby('artist').agg({
        'title': 'count',
        'popularity': 'mean',
        'bpm': 'mean',
        'energy': 'mean',
        'danceability': 'mean'
    }).round(1).reset_index()
    
    artist_stats.columns = ['Artist', 'Number of Songs', 'Avg Popularity', 'Avg BPM', 'Avg Energy', 'Avg Danceability']
    artist_stats = artist_stats.sort_values('Number of Songs', ascending=False)
    
    # Display leaderboard
    st.dataframe(artist_stats.head(10))
    
    # Add bar chart visualization
    st.subheader("Top Artists by Song Count")
    
    top_artists = artist_stats.head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_artists)))
    bars = ax.barh(top_artists['Artist'], top_artists['Number of Songs'], color=colors)
    ax.set_xlabel('Number of Songs')
    ax.set_title('Top 10 Artists by Number of Songs', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Display numbers on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Artist insights
    if len(artist_stats) > 0:
        st.subheader("Artist Insights")
        top_artist = artist_stats.iloc[0]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Most Prolific Artist", top_artist['Artist'])
        with col2:
            st.metric("Their Song Count", top_artist['Number of Songs'])
        with col3:
            st.metric("Avg Popularity", f"{top_artist['Avg Popularity']:.1f}")

# Footer
st.markdown("---")
st.markdown("### 📈 Data Analysis Project")
st.markdown("This interactive dashboard explores music trends and artist performance from 2010-2019.")

# Update and deploy instructions
st.sidebar.markdown("---")
st.sidebar.info("**Deployment Status**: Ready for Streamlit Cloud")
