from sqlalchemy import create_engine, inspect, text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
from openpyxl import load_workbook

def create_table():
    wb = load_workbook("data/LG_IPTV_COL_DEFINITION.xlsx")

    # 활성 시트 선택
    sheet = wb.active

    # 시트 데이터를 데이터프레임으로 변환
    data = sheet.values
    columns = next(data)  # 첫 번째 행을 컬럼명으로 설정
    df = pd.DataFrame(data, columns=columns)
    df = df.iloc[:, 1:5]
    # ['컬럼영문명', '컬럼한글명', '데이터타입', '길이']

    df.rename(columns={
        '컬럼영문명': "col_en", '컬럼한글명': "col_kr", '데이터타입': "dt", '길이': "len"
    }, inplace=True)


    # SQLAlchemy 엔진 생성
    connection_id = 'root'
    connection_pw = '1234'
    # 스키마와 테이블 이름 설정
    schema_name = "ml_mini_project"
    table_name = "lg_iptv"
    
    engine = create_engine(
        "mysql+pymysql://" + connection_id + ":" + connection_pw + "@127.0.0.1:3306/" + schema_name
    )

    # 테이블 존재 여부 확인 (스키마 포함)
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names(schema=schema_name)

    if table_name not in existing_tables:
        # 테이블 생성 쿼리 작성
        create_table_query = f"""
        CREATE TABLE {schema_name}.{table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
        """
        for _, row in df.iterrows():
            col_name = row['col_en']
            data_type = row['dt']
            length = int(row['len']) if row['len'].isdigit() else None

            if data_type == "VARCHAR" and length:
                create_table_query += f"        {col_name} {data_type}({length}),\n"
            else:
                create_table_query += f"        {col_name} {data_type},\n"

        # 테이블 쿼리 마무리
        create_table_query = create_table_query.rstrip(",\n") + "\n    );"

        # 쿼리 실행
        with engine.connect() as conn:
            conn.execute(text(create_table_query))
        print(f"테이블 '{schema_name}.{table_name}'이(가) 성공적으로 생성되었습니다.")
    else:
        print(f"테이블 '{schema_name}.{table_name}'이(가) 이미 존재합니다.")
    
    
    return 0


def save_db():
    # 데이터 읽기
    df = pd.read_csv("data/(sample)LG_IPTV_WTCHNG_STATS_INFO_202107.csv")
    # "https://www.bigdata-culture.kr/bigdata/user/data_market/detail.do?id=612fe7e0-f0d2-11eb-8e60-2bcdc8456bfb"
    
    # 결측치 확인 및 제거
    
    # print("결측치 제거 전:")
    # print(df.isnull().sum())
    # df.dropna(subset=["AVRG_WTCHNG_TIME_CO"], inplace=True)
    # print("결측치 제거 후:")
    # print(df.isnull().sum())

    # SQLAlchemy 엔진 생성
    connection_id = 'root'
    connection_pw = '1234'
    schema_name = "ml_mini_project"
    table_name = "lg_iptv"
    
    engine = create_engine(
        f"mysql+pymysql://{connection_id}:{connection_pw}@127.0.0.1:3306/{schema_name}"
    )

    # 데이터 업로드
    try:
        df.to_sql(
            name=table_name, 
            con=engine, 
            schema=schema_name, 
            if_exists="append", 
            index=False
        )
        print(f"데이터가 '{schema_name}.{table_name}'에 성공적으로 업로드되었습니다.")
    except Exception as e:
        print(f"데이터 업로드 중 오류 발생: {e}")

    return None


def load_db():
    # SQLAlchemy 엔진 생성
    connection_id = 'root'
    connection_pw = '1234'
    # 스키마와 테이블 이름 설정
    schema_name = "ml_mini_project"
    table_name = "lg_iptv"
    
    engine = create_engine(
        "mysql+pymysql://" + connection_id + ":" + connection_pw + "@127.0.0.1:3306/" + schema_name
    )

    data = f"SELECT * FROM {table_name};"
    lg_iptv_data = pd.read_sql(data, con=engine)

    return lg_iptv_data

df = load_db()

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
from imblearn.over_sampling import SMOTE, ADASYN
import joblib
# 전처리
df.drop(["BASE_YM", "AVRG_VOD_PRCHS_PRICE"], axis=1, inplace=True)
df['HLDY_AT'] = np.where(df['HLDY_AT'] == 'Y', 1, 0)
df['CTPRVN_NM'] = df['CTPRVN_NM'].fillna('미응답')
# print(df['CTPRVN_NM'].value_counts())
# CTPRVN_NM
# 경기     151
# 서울     100
# 부산      42
# 경북      41
# 인천      39
# 경남      32
# 대구      29
# 광주      26
# 충남      22
# 강원      21
# 전남      20
# 충북      17
# 전북      17
# 대전      16
# 울산      11
# 제주       8
# 미응답      3


# print(df[df["AVRG_WTCHNG_TIME_CO"].isnull()][['GENRE_LCLAS_NM', 'GENRE_MLSFC_NM', 'HLDY_AT','AVRG_WTCHNG_TIME_CO', 'DAWN_AVRG_WTCHNG_TIME_CO',
#         'AM_AVRG_WTCHNG_TIME_CO', 'PM_AVRG_WTCHNG_TIME_CO',
#         'EVENING_AVRG_WTCHNG_TIME_CO']])


# df_tv = df[df["BRDCST_TY_NM"] == "LINEAR_TV"]


watching_time_cols = ['AVRG_WTCHNG_TIME_CO', 'DAWN_AVRG_WTCHNG_TIME_CO', 'AM_AVRG_WTCHNG_TIME_CO', 'PM_AVRG_WTCHNG_TIME_CO', 'EVENING_AVRG_WTCHNG_TIME_CO']
# print(df[watching_time_cols].describe())
for col in watching_time_cols:
    mean_val = df[col].mean()
    median_val = df[col].median()
    std_val = df[col].std()
    # print(f"결측치 대체 전, {col}, mean: {mean_val}, std: {std_val}")
    df[col] = df[col].fillna(median_val)
    
    # mean_val = df[col].mean()
    # median_val = df[col].median()
    # std_val = df[col].std()
    # print(f"결측치 대체 후, {col}, mean: {mean_val}, std: {std_val}")

# print(df[watching_time_cols].describe())
print(df.isnull().sum())
# 결측치 대체 완료


df_oh_CTPRVN = pd.get_dummies(df["CTPRVN_NM"])
df_oh_CTPRVN.rename(columns={
    '강원': "GW",
    '경기': "GG",
    '경남': "GN",
    '경북': "GB",
    '광주': "GJ",
    '대구': "DG",
    '대전': "DJ",
    '미응답': "NA",
    '부산': "BS",
    '서울': "SE",
    '울산': "US",
    '인천': "IC",
    '전남': "JN",
    '전북': "JB",
    '제주': "JJ",
    '충남': "CN",
    '충북': "CB"
}, inplace=True)
# print(df_oh_CTPRVN.columns)
# ['GW', 'GG', 'GN', 'GB', 'GJ', 'DG', 'DJ', 'NA', 'BS', 'SE', 'US', 'IC', 'JN', 'JB', 'JJ', 'CN', 'CB']

df = pd.concat([df, df_oh_CTPRVN], axis=1)
df.drop(["NA"], axis=1, inplace=True)

# print(len(df_oh_CTPRVN.columns))
# print(df["BRDCST_TY_NM"].value_counts())

# HLDY_AT
# Y    300
# N    295

# BRDCST_TY_NM
# LINEAR_TV    485
# VOD          110
# plt.scatter(df["AM_AVRG_WTCHNG_TIME_CO"].loc[df["BRDCST_TY_NM"]=="LINEAR_TV"], df["PM_AVRG_WTCHNG_TIME_CO"].loc[df["BRDCST_TY_NM"]=="LINEAR_TV"], c="b")
# plt.scatter(df["AM_AVRG_WTCHNG_TIME_CO"].loc[df["BRDCST_TY_NM"]=="VOD"], df["PM_AVRG_WTCHNG_TIME_CO"].loc[df["BRDCST_TY_NM"]=="VOD"], c="r")

# plt.scatter(df["DAWN_AVRG_WTCHNG_TIME_CO"].loc[df["BRDCST_TY_NM"]=="LINEAR_TV"], df["EVENING_AVRG_WTCHNG_TIME_CO"].loc[df["BRDCST_TY_NM"]=="LINEAR_TV"], c="b")
# plt.scatter(df["DAWN_AVRG_WTCHNG_TIME_CO"].loc[df["BRDCST_TY_NM"]=="VOD"], df["EVENING_AVRG_WTCHNG_TIME_CO"].loc[df["BRDCST_TY_NM"]=="VOD"], c="r")
# plt.show()

# print(df.columns)
# ['CTPRVN_NM', 'SIGNGU_NM', 'ADSTRD_NM', 'BRDCST_TY_NM', 'GENRE_LCLAS_NM',
#        'GENRE_MLSFC_NM', 'HLDY_AT', 'AVRG_WTCHNG_CO', 'AVRG_WTCHNG_TIME_CO',
#        'DAWN_AVRG_WTCHNG_TIME_CO', 'AM_AVRG_WTCHNG_TIME_CO',
#        'PM_AVRG_WTCHNG_TIME_CO', 'EVENING_AVRG_WTCHNG_TIME_CO']

# print(df["GENRE_LCLAS_NM"].value_counts())

# 대분류 "방송"은 중분류로 대체
df.loc[df["GENRE_LCLAS_NM"] == "방송", "GENRE_LCLAS_NM"] = df["GENRE_MLSFC_NM"]

df["GENRE_LCLAS_NM"] = df["GENRE_LCLAS_NM"].replace({
    "TV 애니메이션": "애니메이션",
    "애니": "애니메이션",
    "키즈(어린이)": "키즈",
    "iTV등 기타": "기타",
    "공연/음악": "공연음악",
    "시사/교양": "시사교양",
    "연예/오락": "연예오락",
    "TV 드라마": "드라마",
    "해외시리즈": "드라마",
    "정보": "기타",
    "게임": "기타",
})

kids = df["GENRE_LCLAS_NM"] == "키즈"
shopping = df["GENRE_LCLAS_NM"] == "홈쇼핑"
sports = df["GENRE_LCLAS_NM"] == "스포츠"
docu = df["GENRE_LCLAS_NM"] == "다큐"
music = df["GENRE_LCLAS_NM"] == "공연음악"
movie = df["GENRE_LCLAS_NM"] == "영화"
anime = df["GENRE_LCLAS_NM"] == "애니메이션"
drama = df["GENRE_LCLAS_NM"] == "드라마"
enter = df["GENRE_LCLAS_NM"] == "연예오락"
current = df["GENRE_LCLAS_NM"] == "시사교양"
df = df.loc[shopping|sports|anime|kids]
print(df.shape)

le = LabelEncoder()
df["genre_label"] = le.fit_transform(df["GENRE_LCLAS_NM"])

# plt.scatter(df["PM_AVRG_WTCHNG_TIME_CO"], df["EVENING_AVRG_WTCHNG_TIME_CO"], c=df["genre_label"])
# plt.show()

# num_cols = ["HLDY_AT", "DAWN_AVRG_WTCHNG_TIME_CO", "AM_AVRG_WTCHNG_TIME_CO", "PM_AVRG_WTCHNG_TIME_CO", "EVENING_AVRG_WTCHNG_TIME_CO"]
# city_oh_cols = ['GW', 'GG', 'GN', 'GB', 'GJ', 'DG', 'DJ', 'BS', 'SE', 'US', 'IC', 'JN', 'JB', 'JJ', 'CN', 'CB']
num_cols = ["AVRG_WTCHNG_CO", "DAWN_AVRG_WTCHNG_TIME_CO", "AM_AVRG_WTCHNG_TIME_CO", "PM_AVRG_WTCHNG_TIME_CO", "EVENING_AVRG_WTCHNG_TIME_CO"]
# print(df[anime][num_cols].describe())
# night_cols = ["PM_AVRG_WTCHNG_TIME_CO", "EVENING_AVRG_WTCHNG_TIME_CO"]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
# cols = city_oh_cols + num_cols

X = df[num_cols]
y = df["genre_label"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64, stratify=y)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# print(y_train.value_counts())

# smote = SMOTE(random_state=64, k_neighbors=5)
# x_train, y_train = smote.fit_resample(x_train, y_train)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# print(y_train.value_counts())


rf = RandomForestClassifier(random_state=64, max_depth=13, n_estimators=260)

rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
print(rf.feature_importances_)

# GridSearchCV
# rf = RandomForestClassifier(random_state=64)

# params = {
#     "max_depth": range(7, 17),
#     "n_estimators": range(200, 400, 10),
# }

# grid_rf = GridSearchCV(estimator=rf, param_grid=params, cv=3, refit=True)
# grid_rf.fit(x_train, y_train)
# y_pred = grid_rf.predict(x_test)
# print(grid_rf.best_params_)

print(classification_report(y_test, y_pred))
print(le.classes_)


import os

def save_models(le, scaler, rf, model_dir='models'):
    # 모델 파일 저장
    joblib.dump(le, os.path.join(model_dir, 'le.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(rf, os.path.join(model_dir, 'rf.joblib'))
    print("모델들이 저장되었습니다.")

# save_models(le, scaler, rf, "models")

