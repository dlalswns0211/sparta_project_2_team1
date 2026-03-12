# 데이터 분석 심화 프로젝트

## 다이캐스팅 공정 데이터 기반 품질 예측 분석

1조 (위아원)

---

## 프로젝트 개요

다이캐스팅 제조 공정의 센서 및 공정 데이터를 분석하여 불량 여부와 불량 유형을 자동 예측하는 머신러닝 모델 개발

## 분석 목표
1. 다이캐스팅 공정에서 발생하는 다양한 불량 유형(미성형, 박리, 기공, 평탄, 개재물 등)을 자동 예측하는 머신러닝 모델 개발
2. 실시간 품질 예측 체계를 구축하여 조기 경보 시스템 도입 및 불량률 감소

---

## 팀원

| 이름 | 역할 |
|------|------|
| 전재민 | 전처리 & EDA |
| 정서영 | 전처리 & EDA |
| 채지인 | 전처리 & EDA |
| 이민준 | 전처리 & 머신러닝 |
| 이승복 | 전처리 & 머신러닝 |

---

## 프로젝트 구조

```
.
├── data/                           # 원본 데이터셋
│   └── DieCasting_Quality_Raw_Data.csv
├── data_processed/                 # 전처리 완료 데이터셋
│   ├── product_type_1.csv
│   └── product_type_2.csv
├── notebooks/                      # 분석 노트북
│   ├── 1_preprocessing_eda.ipynb   # 전처리 및 EDA
│   ├── 2_ml_type_1.ipynb           # Type 1 머신러닝 모델링
│   └── 2_ml_type_2.ipynb           # Type 2 머신러닝 모델링
├── pyproject.toml
└── README.md
```

---

## 데이터셋

### 원본 데이터: `DieCasting_Quality_Raw_Data.csv`

> 출처: [KAMP (인공지능 제조 플랫폼)](https://www.kamp-ai.kr/aidataDetail?AI_SEARCH=&page=5&DATASET_SEQ=55&EQUIP_SEL=&GUBUN_SEL=&FILE_TYPE_SEL=&WDATE_SEL=)


| 구분 | Product Type 1 | Product Type 2 |
|------|---------------|---------------|
| 데이터 크기 | 2,528행 × 29컬럼 | 1,924행 × 32컬럼 |
| 공정 변수 | 13개 | 14개|
| 센서 변수 | 6개 | 6개 |
| 불량 레이블 | 10개 | 12개 |

### 주요 변수

**공정 변수 (Process)**
- `velocity_1`, `velocity_2`, `velocity_3`, `high_velocity`
- `casting_pressure`, `pressure_rise_time`, `rapid_rise_time`
- `biscuit_thickness`, `clamping_force`, `cycle_time`
- `spray_time`, `spray_1_time`, `spray_2_time`
- `cylinder_pressure` (Type 2만 해당)

**센서 변수 (Sensor)**
- `melting_furnace_temp`, `air_pressure`
- `coolant_temp`, `coolant_pressure`
- `factory_temp`, `factory_humidity`

**불량 레이블 (Defects)**

| 불량 유형 | Type 1 | Type 2 |
|----------|--------|--------|
| 표면 불량 | exfoliation, stain, dent, deformation | stain, dent, scratch, buring_mark |
| 구조 불량 | short_shot, bubble | short_shot, bubble, blow_hole, crack |

### 불량 비율

| 구분 | Type 1 | Type 2 |
|------|--------|--------|
| 표면 불량 | 8.19% | 5.56% |
| 구조 불량 | 15.03% | 20.48% |

---

## 분석 방법론

### 1. 전처리 및 EDA (`1_preprocessing_eda.ipynb`)

**전처리 단계**
1. 데이터 누출(Data Leakage) 방지: `process_id`, `shot`, `product_type` 등 예측 불가 컬럼 제거
2. 결측치 처리 (결측치 없음 확인)
3. 불량 유형을 구조 불량 / 표면 불량 2가지로 그룹화
4. Stratified Train/Test Split (8:2) — 불균형 클래스 비율 유지

| 구분 | 전체 | Train | Test |
|------|------|-------|------|
| Type 1 | 2,528 | 2,022 | 506 |
| Type 2 | 1,924 | 1,539 | 385 |

**EDA 주요 내용**
- 공정 변수 분포 분석 및 이상치 탐색
- 불량 유형별 공정 변수 차이 분석
- Mann–Whitney U 검정을 통한 변수 유의성 검증
- 상관관계 히트맵 분석

---

### 2. 머신러닝 모델링 — 2단계 계층적 분류

불량을 두 단계로 분류하는 **계층적 분류 체계** 적용:

```
Stage 1: 정상 vs 불량 (이진 분류)
           ↓
Stage 2: 정상 / 구조 불량 / 표면 불량 (다중 분류)
```

**사용 모델:** XGBoost (`XGBClassifier`)

**불균형 처리:** `scale_pos_weight`를 이용한 클래스 가중치 조정

---

#### Stage 1: 불량 탐지 (이진 분류)

불량을 놓치지 않는 것을 최우선으로 하여 **재현율(Recall) 최대화** 전략 채택.

| 지표 | Type 1 | Type 2 |
|------|--------|--------|
| Threshold | 0.1 | 0.1 |
| Accuracy | 50% | 49% |
| Precision (불량) | 30% | 32% |
| **Recall (불량)** | **92%** | **92%** |
| F1-Score (불량) | 0.46 | 0.47 |

> 낮은 Threshold를 적용하여 불량 탐지 재현율 92% 달성.
> 제조 공정 특성상 불량 미탐지(False Negative)의 비용이 높으므로 이 전략이 적합.

---

#### Stage 2: 불량 유형 분류 (다중 분류)

Stage 1에서 불량으로 예측된 데이터만 입력으로 사용.

| 지표 | Type 1 | Type 2 |
|------|--------|--------|
| Accuracy | 59% | 55% |
| 정상 F1 | 0.72 | 0.69 |
| 구조 불량 F1 | 0.33 | 0.27 |
| 표면 불량 F1 | 0.17 | 0.21 |

> 표면 불량 데이터가 적어 분류 성능이 상대적으로 낮음.

---

### 주요 특성 중요도 (SHAP)

불량 예측에 일관되게 중요한 변수:

| 순위 | 변수 | 영향 |
|------|------|------|
| 1 | `high_velocity` | 낮을수록 불량 가능성 증가 |
| 2 | `factory_humidity` | 높을수록 불량 확률 증가 |
| 3 | `melting_furnace_temp` | 높을수록 불량 확률 증가 |
| 4 | `casting_pressure` | 구조 불량과 관련 |
| 5 | `coolant_pressure` | 냉각 조건과 관련 |

---

## 주요 결론

1. **2단계 계층적 분류 전략의 유효성:** Stage 1에서 불량/정상 먼저 분리 후 Stage 2에서 유형 분류
2. **핵심 공정 변수 파악:** 사출 속도(`high_velocity`), 공장 습도, 용해로 온도가 품질에 가장 큰 영향
3. **데이터 불균형 극복:** 가중치 조정 및 저 Threshold 전략으로 소수 클래스 탐지율 향상
4. **개선 여지:** 불량 분류 성능 향상을 위한 데이터 수집 및 증강 필요

---

## 사용 기술

| 분류 | 라이브러리 |
|------|-----------|
| 데이터 처리 | `pandas`, `numpy` |
| 머신러닝 | `scikit-learn`, `xgboost`, `imbalanced-learn` |
| 시각화 | `matplotlib`, `seaborn` |
| 통계 검정 | `statsmodels` |
| 환경 관리 | `uv`, `Jupyter Notebook` |

---

## 실행 방법

```bash
# 의존성 설치
uv sync

# 분석 순서
# 1. 전처리 및 EDA
#    src/1_preprocessing_eda.ipynb

# 2. 머신러닝 모델링
#    src/2_ml_type_1.ipynb  (Product Type 1)
#    src/2_ml_type_2.ipynb  (Product Type 2)
```
