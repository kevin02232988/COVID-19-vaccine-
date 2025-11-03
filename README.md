![header](https://capsule-render.vercel.app/api?type=soft&color=auto&height=300&section=header&text=vaccine%20Review💉&fontSize=90)<br/>
## MobileBert를 활용한 코로나 백 리뷰 감성분석 프로젝트
<img src="https://img.shields.io/badge/PyTorch-E34F26?style=flat-square&logo=PyTorch&logoColor=white"/></a>
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/></a>
# COVID Vaccine Controversy Analysis by BERT 🦠
Multilingual BERT를 활용한 코로나 백신 여론 분석 프로젝트

[![PyTorch](https://img.shields.io/badge/PyTorch-E34F26?style=flat-square&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21C?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white)](https://www.python.org/)

---

## 1. 개요 및 목표 (Overview & Thesis)

### 1-1. 문제 정의 및 프로젝트의 의의
본 프로젝트는 COVID-19 팬데믹 기간 동안 온라인 커뮤니티(Reddit)에 나타난 감정적 양극화를 BERT 모델 기반의 감성 분석으로 정량화합니다.

- **핵심 목표**: 98,277건의 데이터를 분석하여 사회적 의무(Mandate) 관련 논란이 과학적 부작용보다 여론 확산에 미친 영향이 더 컸음을 입증
- **성공 지표**: 9,827건의 수동 라벨링 데이터를 기반으로 F1 Score 0.6438 달성 → 모델 신뢰성 검증

### 1-2. 프로젝트 성공 지표 (Performance Metrics)

| 지표 | 결과 값 | 통찰 (보고서 성공 지표) |
|------|---------|------------------------|
| Accuracy (정확도) | 0.8204 (82.04%) | 전반적인 감정 예측 능력이 우수함 |
| F1 Score (균형 점수) | 0.6438 (64.38%) | 74% 부정 편향 데이터에서 긍/부정 클래스를 균형 있게 분류했음을 입증 |

---

## 2. 데이터 수집 및 정제 과정 (극복의 스토리)

### 2-1. 데이터 확보 난관 및 최종 소스
본 프로젝트는  Naver, WebMD, Drugs.com 등 **6가지 크롤링/API 시도(IP 차단 문제)**를 극복하고, Reddit API를 통해 98,277건의 고품질 데이터를 확보했습니다.

## 데이터 수집 시도 및 결과 요약

| 소스 | 언어 | 수집 내용 | 결과/문제점 |
|------|------|-----------|-------------|
| **Naver 지식iN** | 한국어 | Q&A (타이레놀/피임약) | 성공: 답변 40,000건 확보함. 단, 백신과는 무관한 내용이 너무 많아 사용하기 힘들어보이고 질문 내용은 확인이 불가능하고 답변 내용만 확인이 가능해서 데이터로서 적합하지 않음|
| **Naver News 댓글** | 한국어 | 뉴스 댓글 감성 분석 | 실패: Selenium CSS 선택자 불일치 |
| **DC Inside** | 한국어 | 포럼 댓글 감성 분석 | 성공: 답변 20,000건 확보. 단, 비속어 및 백신과는 무관한 내용이 너무 많아 사용하기 힘들어보임 |
| **Reddit API (PRAW)** | 영어 | 댓글/게시글 수집 (Top Posts) | 성공: 52,827건 확보 (최종 데이터) |
| **Pushshift API** | 영어 | 대량 과거 데이터 | 실패: 403 Forbidden (서버 차단). 약 1000 건 정도만 확보함| 
| **Drugs.com** | 영어 | 전문 리뷰 | 실패: 403 IP 차단 및 Google Cache 우회 실패. 약 500 건 정도만 확보함 |
| **WebMD** | 영어 | 포럼 댓글 | 실패: 403 IP 차단. 약 6000 건 정도만 확보함 | 
| **HealthBoards** | 영어 | 포럼 댓글 | 실패: 404 URL 오류. 약 3000 건 정도만 확보함 |
| **Patient.info** | 영어 | 리뷰/포럼 | 실패: Requests 차단 취약. 약 400건 정도만 확보함 |


- **최종 소스**: Reddit과 WebMD, Pushshift, HealthBoards, Patient.info, Drugs.com을 모두 통합함 (r/Coronavirus, r/vaccine 등 Top Posts)
- **최종 규모**: 98,277건의 영문 게시글 및 댓글

### 2-2. 엄격한 데이터 전처리 과정 (정확도 82%의 근거)
BERT 모델의 고정확도를 위해 다음 3단계 필터링 적용:

1. **노이즈 및 삭제 항목 제거**: 4단어 미만 텍스트, API 수집 중 발생하는 `[No Content]`, `[deleted]` 등 무의미한 플레이스홀더 모두 제거
2. **언어 및 특수문자 필터링**: 영어 이외의 언어 비율 10% 초과 데이터, 40% 이상 특수문자로 구성된 노이즈 행 제거 → 영문 데이터 순도 향상
3. **라벨 정규화**: 수동 라벨을 BERT가 인식하도록 **Positive**와 **Negative**로 정확히 매핑

---

## 3. 핵심 분석 시각화 및 결론 (Key Visualization)

### 3-1. 전체 온라인 여론 분포 (Figure 1)
온라인 여론이 부정적 감정에 의해 3:1 비율로 지배됨을 입증

- **감정비율 (%)**
  - Negative (부정): 76.27%
  - Positive (긍정): 23.73%

### 3-2. 부정 여론의 핵심 논란 키워드 (Figure 3)
74,954건의 부정 텍스트 분석 결과, 논란의 초점은 과학이 아닌 사회적 통제에 있었음이 드러남

| 순위 | 키워드 | 언급 횟수 | 보고서 통찰 (사회적 논란의 근거) |
|------|--------|-----------|----------------------------------|
| 1 | mask / masks | 8,736회 | 방역 의무와 관련된 사회적 통제에 대한 반발이 가장 큼 |
| 2 | work | 4,868회 | 고용과 직장을 연계한 백신 의무화에 대한 불만 주요 논란 |
| 3 | right | 4,253회 | 개인의 자유와 권리 침해 주장이 부정 여론의 핵심 동력 |
