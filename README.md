## 📄 README.md 초안 – 방송 장르 예측 챗봇 (Streamlit + Scikit-learn)
---
![image](thumbnails/streamlit_app.png)
---

### 📌 프로젝트 소개

이 프로젝트는 사용자의 TV 시청 패턴을 입력 받아, **방송 장르(스포츠, 애니메이션, 키즈, 홈쇼핑)** 를 예측하는 **머신러닝 기반 챗봇**입니다.
사용자는 슬라이더로 시청 시간을 입력하면, Scikit-learn으로 학습된 **Random Forest 모델**이 방송 장르를 예측하고 챗봇 형식으로 결과를 반환합니다.

---

### 🛠️ 기술 스택

* **Frontend**: [Streamlit](https://streamlit.io/)
* **Backend**: Python 3.12
* **ML 모델**: Scikit-learn (`RandomForestClassifier`, `LabelEncoder`, `StandardScaler`)
* **모델 저장/로딩**: `joblib`
* **이미지 출력**: 각 예측 장르별 시각적 응답 이미지 (`img/` 폴더 활용)

---

### 📁 프로젝트 구조

```
genre_chatbot/
├── app.py                # Streamlit 메인 앱
├── models/
│   ├── rf.joblib         # RandomForest 모델
│   ├── scaler.joblib     # StandardScaler
│   └── le.joblib         # LabelEncoder
├── img/
│   ├── 0.png             # 스포츠 예측 결과 이미지
│   ├── 1.png             # 애니메이션 예측 결과 이미지
│   ├── 2.png             # 키즈 예측 결과 이미지
│   └── 3.png             # 홈쇼핑 예측 결과 이미지
└── README.md             # 설명 문서 (본 파일)
```

---

### 🚀 실행 방법

#### 1. 가상 환경 준비 및 의존성 설치

```bash
conda create -n genre_chatbot python=3.12
conda activate genre_chatbot
pip install streamlit scikit-learn joblib
```

#### 2. 실행

```bash
streamlit run app.py
```

---

### 🧠 입력 설명

사용자는 다음 다섯 가지 슬라이더 입력을 통해 예측을 수행합니다:

* **하루 평균 시청 프로그램 수** (`avrg_wtchng_co`)
* **새벽 평균 시청 시간 (분)** (`dawn_avrg_wtchng_time_co`)
* **오전 평균 시청 시간 (분)** (`am_avrg_wtchng_time_co`)
* **오후 평균 시청 시간 (분)** (`pm_avrg_wtchng_time_co`)
* **저녁 평균 시청 시간 (분)** (`evening_avrg_wtchng_time_co`)

---

### 💬 사용 예시

1. 슬라이더로 시청 데이터를 설정하고 **\[채팅 추가]** 버튼 클릭
2. 사용자 입력을 바탕으로 방송 장르 예측
3. 챗봇 응답 메시지와 함께 시각적 이미지 출력
4. 모든 채팅 히스토리는 우측 영역에 누적 표시

---

### 🧪 모델 훈련 및 저장 예시 (선택 사항)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# 예시 데이터 학습
X = ...  # feature matrix
y = ...  # target labels

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier()
rf.fit(X_scaled, y_encoded)

# 모델 저장
joblib.dump(le, 'models/le.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(rf, 'models/rf.joblib')
```

---

### 📌 주요 기능 요약

* ✅ Streamlit 기반 웹 UI
* ✅ 시청 패턴 입력 → 방송 장르 예측
* ✅ 대화 저장 및 시각적 응답 제공
* ✅ 채팅 내역 초기화 기능 포함
* ✅ 학습된 모델 재사용 (joblib)

---

### 📷 예측 결과 이미지 예시

| 장르    | 이미지 파일명 |
| ----- | ------- |
| 스포츠   | `0.png` |
| 애니메이션 | `1.png` |
| 키즈    | `2.png` |
| 홈쇼핑   | `3.png` |

---

### 📊 사용 데이터셋 소개

이 프로젝트는 **LG유플러스의 'U+ IPTV 시청통계' 무료 샘플 데이터**를 기반으로 머신러닝 모델을 학습하였습니다.

- **데이터셋 이름**: U+ IPTV 시청통계
- **제공처**: LG유플러스 / 한국지능정보사회진흥원(NIA)
- **주요 내용**: IPTV 가입자들의 시간대별 평균 시청시간, 시청 프로그램 수, 시청 장르 등
- **활용 목적**: 시청 패턴 데이터를 기반으로 사용자가 선호할 방송 장르를 예측하는 챗봇 구현
- **URL**:  
  [https://www.bigdata-culture.kr/bigdata/user/data_market/detail.do?id=612fe7e0-f0d2-11eb-8e60-2bcdc8456bfb](https://www.bigdata-culture.kr/bigdata/user/data_market/detail.do?id=612fe7e0-f0d2-11eb-8e60-2bcdc8456bfb)
