import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置页面
st.set_page_config(page_title="Billboard音乐分析", layout="wide")

# 标题
st.title("🎵 2010-2019年热门歌曲分析")
st.write("探索过去十年热门歌曲的特征和趋势")

# 创建完整的示例数据集
@st.cache_data
def load_data():
    # 基于真实Billboard数据特征的示例数据
    np.random.seed(42)
    years = list(range(2010, 2020))
    
    data = []
    for year in years:
        for i in range(60):  # 每年60首热门歌曲
            song_id = f"{year}_{i+1}"
            
            # 生成逼真的音乐数据
            bpm = np.random.randint(80, 180)
            energy = np.random.randint(50, 100)
            danceability = np.random.randint(40, 95)
            popularity = np.random.randint(70, 100)
            
            # 模拟音乐特征趋势（随时间变化）
            if year >= 2015:
                bpm += 10  # 近年音乐节奏更快
                energy += 5
            
            # 艺术家和流派
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
                'duration_ms': np.random.randint(180000, 240000)  # 3-4分钟
            })
    
    return pd.DataFrame(data)

df = load_data()


# 侧边栏 - 数据过滤
st.sidebar.header("🎛️ 数据过滤")

# 年份选择
selected_years = st.sidebar.slider(
    "选择年份范围",
    min_value=int(df['year'].min()),
    max_value=int(df['year'].max()),
    value=(2015, 2019)
)

# 流派选择
all_genres = ['全部'] + sorted(df['genre'].unique().tolist())
selected_genre = st.sidebar.selectbox("选择音乐流派", all_genres)

# 应用过滤
filtered_df = df[
    (df['year'] >= selected_years[0]) & 
    (df['year'] <= selected_years[1])
]

if selected_genre != '全部':
    filtered_df = filtered_df[filtered_df['genre'] == selected_genre]

# 主显示区域
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("总歌曲数", len(filtered_df))
with col2:
    st.metric("年份范围", f"{filtered_df['year'].min()}-{filtered_df['year'].max()}")
with col3:
    st.metric("艺术家数量", filtered_df['artist'].nunique())
with col4:
    st.metric("流派数量", filtered_df['genre'].nunique())

# 标签页布局
tab1, tab2, tab3, tab4 = st.tabs(["📊 数据概览", "📈 趋势分析", "🎵 音乐特征", "🏆 排行榜"])

with tab1:
    st.subheader("数据预览")
    st.dataframe(filtered_df.head(10))
    
    # 基本统计
    st.subheader("数据统计")
    st.write(filtered_df[['bpm', 'energy', 'danceability', 'popularity']].describe())

with tab2:
    st.subheader("音乐特征年度趋势")
    
    # 按年份分组计算平均值
    yearly_avg = filtered_df.groupby('year').agg({
        'bpm': 'mean',
        'energy': 'mean', 
        'danceability': 'mean',
        'popularity': 'mean'
    }).reset_index()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # BPM趋势
    ax1.plot(yearly_avg['year'], yearly_avg['bpm'], marker='o', linewidth=2, markersize=8)
    ax1.set_title('平均BPM趋势')
    ax1.set_xlabel('年份')
    ax1.set_ylabel('BPM')
    ax1.grid(True, alpha=0.3)
    
    # 能量趋势
    ax2.plot(yearly_avg['year'], yearly_avg['energy'], marker='s', color='orange', linewidth=2, markersize=8)
    ax2.set_title('平均能量趋势')
    ax2.set_xlabel('年份')
    ax2.set_ylabel('能量')
    ax2.grid(True, alpha=0.3)
    
    # 舞蹈性趋势
    ax3.plot(yearly_avg['year'], yearly_avg['danceability'], marker='^', color='green', linewidth=2, markersize=8)
    ax3.set_title('平均舞蹈性趋势')
    ax3.set_xlabel('年份')
    ax3.set_ylabel('舞蹈性')
    ax3.grid(True, alpha=0.3)
    
    # 流行度趋势
    ax4.plot(yearly_avg['year'], yearly_avg['popularity'], marker='d', color='red', linewidth=2, markersize=8)
    ax4.set_title('平均流行度趋势')
    ax4.set_xlabel('年份')
    ax4.set_ylabel('流行度')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("音乐特征分布")
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature = st.selectbox("选择特征", ['bpm', 'energy', 'danceability', 'popularity'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(filtered_df[feature], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel(feature)
        ax.set_ylabel('歌曲数量')
        ax.set_title(f'{feature} 分布')
        st.pyplot(fig)
    
    with col2:
        # 流派分布
        genre_counts = filtered_df['genre'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('音乐流派分布')
        st.pyplot(fig)

with tab4:
    st.subheader("热门艺术家排行榜")
    
    # 按艺术家统计
    artist_stats = filtered_df.groupby('artist').agg({
        'title': 'count',
        'popularity': 'mean',
        'bpm': 'mean'
    }).round(1).reset_index()
    artist_stats.columns = ['艺术家', '歌曲数量', '平均流行度', '平均BPM']
    artist_stats = artist_stats.sort_values('歌曲数量', ascending=False)
    
    st.dataframe(artist_stats.head(10))
