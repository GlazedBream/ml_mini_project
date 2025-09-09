import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# .env 파일에서 환경 변수 로드
load_dotenv()

# DB 접속 정보 가져오기
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_SCHEMA = os.getenv("DB_SCHEMA")
DB_TABLE = os.getenv("DB_TABLE")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def get_engine():
    """SQLAlchemy 엔진을 생성하여 반환"""
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_SCHEMA]):
        raise ValueError("데이터베이스 연결 정보가 .env 파일에 올바르게 설정되지 않았습니다.")
    return create_engine(
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_SCHEMA}"
    )

def load_data_from_db():
    """데이터베이스에서 시청 데이터를 로드합니다."""
    print("DB에서 데이터를 로드합니다...")
    try:
        engine = get_engine()
        query = f"SELECT * FROM {DB_TABLE}"
        df = pd.read_sql(query, engine)
        print(f"{len(df)}개의 레코드를 성공적으로 로드했습니다.")
        return df
    except Exception as e:
        print(f"DB 데이터 로딩 중 오류 발생: {e}")
        return pd.DataFrame()

def preprocess_for_segmentation(df):
    """지역별 세분화를 위해 데이터를 전처리합니다."""
    numeric_cols = ['AVRG_WTCHNG_CO', 'DAWN_AVRG_WTCHNG_TIME_CO', 'AM_AVRG_WTCHNG_TIME_CO', 'PM_AVRG_WTCHNG_TIME_CO', 'EVENING_AVRG_WTCHNG_TIME_CO']
    region_df = df.groupby(['CTPRVN_NM', 'SIGNGU_NM', 'ADSTRD_NM'])[numeric_cols].mean().reset_index()
    for col in numeric_cols:
        median_val = region_df[col].median()
        region_df[col].fillna(median_val, inplace=True)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(region_df[numeric_cols])
    print("세분화를 위한 데이터 전처리를 완료했습니다.")
    return region_df, scaled_features, numeric_cols, scaler

def find_optimal_clusters(scaled_features, max_k=10):
    """Elbow Method 차트를 생성하고 파일로 저장한 뒤 경로를 반환합니다."""
    print("최적의 클러스터 개수를 찾고 있습니다 (Elbow Method)...")
    iters = range(2, max_k + 1)
    sse = []
    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('클러스터 수 (K)')
    ax.set_ylabel('SSE (Sum of Squared Errors)')
    ax.set_title('최적 K를 위한 Elbow Method')
    plt.grid(True)
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    chart_path = f'result/elbow_method_chart_{timestamp}.png'
    plt.savefig(chart_path, bbox_inches='tight')
    print(f"Elbow Method 차트를 '{chart_path}'에 저장했습니다.")
    plt.close(fig) # 메모리 누수 방지를 위해 fig 객체를 닫음
    return chart_path

def create_radar_chart(cluster_summary, scaler):
    """클러스터별 특성을 담은 레이더 차트를 생성하고 파일로 저장한 뒤 경로를 반환합니다."""
    original_centers = scaler.inverse_transform(cluster_summary)
    cluster_summary_orig = pd.DataFrame(original_centers, columns=cluster_summary.columns, index=cluster_summary.index)

    labels = cluster_summary_orig.columns
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for i, row in cluster_summary_orig.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, label=f'Cluster {i}')
        ax.fill(angles, values, alpha=0.25)
        
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("클러스터별 시청 패턴 (Radar Chart)", size=20, y=1.1)
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    chart_path = f'result/radar_chart_{timestamp}.png'
    plt.savefig(chart_path, bbox_inches='tight')
    print(f"Radar 차트를 '{chart_path}'에 저장했습니다.")
    plt.close(fig) # 메모리 누수 방지를 위해 fig 객체를 닫음
    return chart_path

def interpret_clusters(cluster_summary):
    """클러스터별 특성을 해석하여 페르소나를 부여합니다."""
    personas = {}
    time_cols = ['DAWN_AVRG_WTCHNG_TIME_CO', 'AM_AVRG_WTCHNG_TIME_CO', 'PM_AVRG_WTCHNG_TIME_CO', 'EVENING_AVRG_WTCHNG_TIME_CO']
    col_map = {
        'DAWN_AVRG_WTCHNG_TIME_CO': '새벽',
        'AM_AVRG_WTCHNG_TIME_CO': '오전',
        'PM_AVRG_WTCHNG_TIME_CO': '오후',
        'EVENING_AVRG_WTCHNG_TIME_CO': '저녁'
    }

    for i, row in cluster_summary.iterrows():
        dominant_time_col = row[time_cols].idxmax()
        dominant_time = col_map[dominant_time_col]
        
        if row['AVRG_WTCHNG_CO'] > cluster_summary['AVRG_WTCHNG_CO'].mean():
            persona = f'{dominant_time} 시청이 활발한 다(多)시청 그룹'
        else:
            persona = f'{dominant_time} 시청 중심의 소(小)시청 그룹'
        personas[i] = persona
    return personas

def run_segmentation(df, n_clusters=4):
    """K-Means 클러스터링을 수행하고 분석 결과와 차트 파일 경로를 반환합니다."""
    region_df, scaled_features, numeric_cols, scaler = preprocess_for_segmentation(df)
    
    print(f"{n_clusters}개의 클러스터로 K-Means 분석을 수행합니다...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    region_df['cluster'] = kmeans.fit_predict(scaled_features)
    
    cluster_summary_scaled = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols)
    
    radar_chart_path = create_radar_chart(cluster_summary_scaled, scaler)
    personas = interpret_clusters(cluster_summary_scaled)
    
    print("K-Means 클러스터링 및 시각화, 해석을 완료했습니다.")
    return region_df, personas, radar_chart_path
