import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
import pandas as pd
from openpyxl import load_workbook

# .env 파일에서 환경 변수 로드
load_dotenv()

# DB 접속 정보 가져오기
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_SCHEMA = os.getenv("DB_SCHEMA")
DB_TABLE = os.getenv("DB_TABLE")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

def get_engine():
    """SQLAlchemy 엔진을 생성하여 반환"""
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_SCHEMA]):
        raise ValueError("데이터베이스 연결 정보가 .env 파일에 올바르게 설정되지 않았습니다.")
    return create_engine(
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_SCHEMA}"
    )

def create_table():
    """data/LG_IPTV_COL_DEFINITION.xlsx 파일을 읽어 DB에 테이블을 생성합니다."""
    engine = get_engine()
    print(f"테이블 생성을 시도합니다: '{DB_SCHEMA}.{DB_TABLE}'")
    try:
        wb = load_workbook("data/LG_IPTV_COL_DEFINITION.xlsx")
        sheet = wb.active
        data = sheet.values
        columns = next(data)
        df = pd.DataFrame(data, columns=columns)
        df = df.iloc[:, 1:5]
        df.rename(columns={
            '컬럼영문명': "col_en", '컬럼한글명': "col_kr", '데이터타입': "dt", '길이': "len"
        }, inplace=True)

        inspector = inspect(engine)
        if DB_TABLE not in inspector.get_table_names(schema=DB_SCHEMA):
            create_table_query = f"CREATE TABLE {DB_SCHEMA}.{DB_TABLE} (id INT AUTO_INCREMENT PRIMARY KEY, "
            for _, row in df.iterrows():
                col_name = row['col_en']
                data_type = row['dt']
                length = int(row['len']) if str(row['len']).isdigit() else None
                if data_type == "VARCHAR" and length:
                    create_table_query += f"{col_name} {data_type}({length}), "
                else:
                    create_table_query += f"{col_name} {data_type}, "
            create_table_query = create_table_query.rstrip(", ") + ");"

            with engine.connect() as conn:
                conn.execute(text(create_table_query))
            print(f"테이블 '{DB_SCHEMA}.{DB_TABLE}'이(가) 성공적으로 생성되었습니다.")
        else:
            print(f"테이블 '{DB_SCHEMA}.{DB_TABLE}'이(가) 이미 존재합니다.")
    except Exception as e:
        print(f"테이블 생성 중 오류 발생: {e}")

def save_db():
    """CSV 데이터를 읽어 DB에 저장합니다."""
    engine = get_engine()
    print(f"데이터 적재를 시도합니다: '{DB_SCHEMA}.{DB_TABLE}'")
    try:
        # 데이터가 이미 있는지 확인
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {DB_SCHEMA}.{DB_TABLE}")).scalar()
        if result > 0:
            print(f"테이블 '{DB_SCHEMA}.{DB_TABLE}'에 이미 데이터가 존재합니다. 데이터 삽입을 건너뜁니다.")
            return

        # 데이터 읽기
        df = pd.read_csv("data/(sample)LG_IPTV_WTCHNG_STATS_INFO_202107.csv")
        
        # 데이터 업로드
        df.to_sql(
            name=DB_TABLE, 
            con=engine, 
            schema=DB_SCHEMA, 
            if_exists="append", 
            index=False
        )
        print(f"데이터 {len(df)}건이 '{DB_SCHEMA}.{DB_TABLE}'에 성공적으로 업로드되었습니다.")
    except Exception as e:
        print(f"데이터 업로드 중 오류 발생: {e}")

def main():
    """DB 테이블 생성 및 데이터 적재를 수행합니다."""
    print("데이터베이스 작업을 시작합니다...")
    create_table()
    save_db()
    print("데이터베이스 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()