![header](https://capsule-render.vercel.app/api?type=soft&color=auto&height=300&section=header&text=vaccine%20Review💉&fontSize=90)

# 🦠 COVID Vaccine Controversy Analysis by BERT
**Multilingual BERT를 활용한 코로나 백신 여론 분석 프로젝트**

[![PyTorch](https://img.shields.io/badge/PyTorch-E34F26?style=flat-square&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21C?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white)](https://www.python.org/)

---

##  1. 개요 및 목표 (Overview & Thesis)

###  문제 정의 및 프로젝트의 의의
본 프로젝트는 COVID-19 팬데믹이 지나간 현재 시점에서,  
그동안 온라인 커뮤니티에 남겨진 댓글과 리뷰를 수집을 한 후 필요없는 데이터나 가비지 데이터를 정제하여 유의미한 정보만을 남긱고 그것을 분석해 보고 어떤 새로운 결론을 찾는 것을을 목표로 합니다.


- **핵심 목표**: 약 **11만 건의 리뷰 데이터**를 분석해  
  사회적 의무(Mandate) 논란이 **과학적 부작용보다 여론 확산에 더 큰 영향을 미쳤음**을 입증  
- **성공 지표**: 2만 건의 수동 라벨링 데이터 기반 **F1 Score = 0.6438** 달성 → 모델 신뢰성 확보

---

##  2. 데이터 수집 및 정제 과정 (극복의 스토리)

### 2-1. 데이터 확보 난관 및 최종 소스

> 본 프로젝트는 Naver, WebMD, Drugs.com 등 **6가지 크롤링 및 API 시도**를 거쳐  
> **Reddit API**를 중심으로 **98,277건의 고품질 데이터**를 확보했습니다.

| 소스 | 언어 | 수집 내용 | 결과 / 문제점 |
|------|------|-----------|----------------|
| **Naver 지식iN** | 🇰🇷 | Q&A (타이레놀/피임약) | 답변 32,000건 확보. 단, 백신 관련 내용 부족 및 질문 본문 확인 불가 |
| **Naver News 댓글** | 🇰🇷 | 뉴스 댓글 감성 분석 | 실패: Selenium CSS 선택자 불일치 |
| **DC Inside** | 🇰🇷 | 포럼 댓글 감성 분석 | 23,000건 확보. 비속어 및 백신 무관한 내용 다수 |
| **Reddit API (PRAW)** | 🇺🇸 | 댓글/게시글 (Top Posts) | ✅ 72,827건 확보 (핵심 데이터) |
| **Pushshift API** | 🇺🇸 | 과거 Reddit 데이터 | ❌ 403 Forbidden (서버 차단), 약 2,100건 확보 |
| **Drugs.com** | 🇺🇸 | 전문 리뷰 | ❌ 403 IP 차단, 약 1500건 확보 |
| **WebMD** | 🇺🇸 | 포럼 댓글 | ❌ 403 IP 차단, 약 16,800건 확보 |
| **HealthBoards** | 🇺🇸 | 포럼 댓글 | ❌ 404 오류, 약 3,100건 확보 |
| **Patient.info** | 🇺🇸 | 리뷰/포럼 | ❌ Requests 차단, 약 480건 확보 |

**최종 데이터 통합**
-  Reddit + WebMD + Pushshift + HealthBoards + Patient.info + Drugs.com 통합
-  총 **약 12만 건** (영문 및 한글 포함)
-  이후 **한국어 데이터 제거 → 영어 데이터만 남김**

-  최종 데이터: 약 10만8천건 (영문만 있음)

---

##  3. 데이터 전처리 파이프라인 (4단계)

###  단계 1: 구조적 노이즈 제거 (API Placeholder)

| 가비지 유형 | 문제 원인 | 적용 기법 | 처리 내용 |
|:-------------|:------------|:------------|:------------|
| `[deleted]`, `[No Content]` | Reddit API 삭제 게시물 | `pandas filtering` | 해당 문자열 포함 행 제거 |
| 짧은 잡담 (`lol`, `ok`) | 의미 없는 짧은 댓글 | `length check` | 20자 미만 / 5단어 미만 제거 |

---

### 🔹 단계 2: 언어적 노이즈 제거 (한국어 분리)

| 가비지 유형 | 문제 원인 | 적용 기법 | 처리 내용 |
|:-------------|:------------|:------------|:------------|
| 한글 포함 데이터 | Naver 크롤링 데이터 | `RegEx filtering` | 한글 비율 10% 초과 시 제거 |

---

###  단계 3: 형식적 노이즈 제거 (특수문자, URL 등)

| 가비지 유형 | 문제 원인 | 적용 기법 | 처리 내용 |
|:-------------|:------------|:------------|:------------|
| 특수문자/URL | 이모티콘, 반복 기호 | `RegEx ratio check` | 알파벳 외 문자 비율 40% 초과 제거 |
| 불용어 | the, a, is, covid, vaccine 등 | `Stopword list` | 핵심 키워드(`mask`, `mandate`) 중심으로 토픽 모델링 효율 개선 |

---

이 과정들을 통해 이전 데이터: 약 10만8천건  ---> 9만 7천건까지 정제하였다.


이후 전체 데이터에서 10% 샘플을 뽑아 긍/부정으로 직접 라벨링을 진행했고  3.5:7.5 의비율로 긍/부정이 구분되었다. 
추가로 직접적으로 백신과 코로나, 부작용에 대한 언급은 얼마나 있는지 확인하기 위해 10% 셈플에 백신과 코로나, 부작용에 대한 언급이 있으면 True. 없다면 false로 추가로 라벨링을 진행하였다. 그 결과는 False=84% True=16의 결과가 나왔다.
이후 KoELECTRA를 사용하여 러신러닝을 진행하여 전체 데이터의 긍부정 비율과 true faluse 비율을 얻었다.
 1. 전체 감정 분포 ---
| Predicted_Sentiment   |   Count |   Ratio (%) |
|:----------------------|--------:|------------:|
| Negative              |   74954 |       76.27 |
| Positive              |   23323 |       23.73 |


2. True/false 비율
related_to_vaccine
False   83.29%
True    16.71%


3.모델 최종 성능 ---
✅ Accuracy (정확도): 0.8204
✅ F1 Score (균형 점수): 0.6438

이라는 결과가 나왔다 

시계열 데이터를 따라서 시간이 변함에 따라 부정의 비율이 어떻게 나오는지 도표를 통해서 확인해 봤다.

##  4. 토픽 모델링 (Topic Modeling)

### 코드 요약

| 단계 | 주요 내용 | 사용 라이브러리 / 함수 | 목적 |
|------|------------|--------------------------|------|
| 1️⃣ 데이터 로드 | `Real_Final.csv` 불러오기 | `pandas.read_csv()` | 데이터 준비 |
| 2️⃣ 전처리 | URL, 특수문자 제거 + 토큰화 + 표제어화 | `re`, `nltk` | 노이즈 제거 |
| 3️⃣ DTM 생성 | BoW 변환 및 희귀 단어 제거 | `gensim.Dictionary`, `doc2bow()` | 텍스트 수치화 |
| 4️⃣ LDA 학습 | `num_topics=10`, `passes=20` | `gensim.models.LdaModel` | 주요 주제 추출 |
| 5️⃣ 결과 해석 | 각 토픽 상위 10개 단어 분석 | `lda_model.print_topics()` | 핵심 주제 파악 |

---

###  LDA 토픽 모델링 결과 (K=10)

| 토픽 번호 | 주요 키워드 | 해석된 주제 | 핵심 논의 내용 |
|------------|--------------|--------------|----------------|
| **1** | state, government, school, free, business | 국가 정책 및 지역 행정 | 정부 정책, 학교 운영, 사업 규제 등 |
| **2** | people, doctor, health, risk, vaccination | 일반인 건강 및 의료 접근성 | 개인 건강, 접종 필요성, 위험 인식 |
| **3** | company, money, debt, share, loan | 경제적 영향 및 기업 금융 | 코로나가 금융·기업에 미친 영향 |
| **4** | like, dont, think, time, year | 개인의 생각 및 감정 표현 | 일상적 감정, 의견 공유 중심 |
| **5** | mask, wear, face, protect, approved | 마스크 착용 및 방역 조치 | 마스크 의무화, 보호 장비 논의 |
| **6** | side, country, trump, american, world | 정치적 갈등 및 국가 상황 | 미국 중심의 정치·사회적 갈등 |
| **7** | insurance, cost, headache, paid, market | 비용 및 보험 문제 | 의료비, 보험, 시장 변동성 |
| **8** | vaccine, covid, effect, virus, booster | 백신 효능 및 과학적 논의 | 백신 효과, 바이러스, 부스터샷 |
| **9** | week, work, month, sick, day | 일상 및 근무 환경 변화 | 재택근무, 격리 등 생활 패턴 변화 |
| **10** | post, read, comment, source, link | 정보 공유 및 커뮤니티 소통 | 게시글·링크를 통한 정보 교환 |

---

###  종합 시사점
- **토픽 3 & 7**: 경제적 영향(기업 vs. 개인 비용)이 주요 논의 축  
- **토픽 5 & 8**: 방역 조치 및 백신 효능 관련 과학적 논의  
- **토픽 6**: 정치적 분열과 국가적 감정이 백신 담론에 영향  
- **토픽 10**: 신뢰할 수 있는 정보 출처에 대한 사회적 갈망 반영


## 결론과 한계점 
- **사회적 논란의 핵심 축**은 과학적 부작용보다 **정치·경제·사회적 이슈**에 집중.
- **LDA와 빈도 분석** 결과가 서로 일치하며, **마스크·백신·정부정책·경제영향**이 공통 핵심 키워드로 등장함 
- 이는 코로나 백신 논란이 단순한 의학적 문제를 넘어, **사회적 신뢰와 정책적 갈등의 문제**였음을 시사한다.
따라서 일반적으로 사회적 신뢰와 정책적 갈등이 심할때 사회 불안지수, 경재 불안지수가 높아지기에 두 그래프를 X을 시간을 기준으로 비교해 보기로 했다
시계열 데이터를 따라서 시간이 변함에 따라 부정의 비율이 어떻게 나오는지 도표를 통해서 확인해 봤다.



## 두 그래프 비교 결과
피어슨 상관계수 = -0.006
DTW 거리 = 1280
참으로 실망적인 결과였다, 



피어슨 상관계수 ≈ -0.006은  0에 매우 가까움 → 두 그래프의 선 모양은 거의 직선적으로 상관이 없음.

즉, 한 그래프가 올라가면 다른 그래프가 올라가거나 내려가는 경향이 거의 없다는 의미입니다.


DTW(Dynamic Time Warping)는 두 시계열의 모양 차이를 측정하는 지표인데, 값이 크면 패턴이 많이 다름을 의미합니다.

1280 정도면 두 그래프의 모양이 상당히 다르다는 것을 보여줍니다.


- 하지만, 코로나 펜데믹이 이제는 과거가된 시점에서 이 결론은 너무나 naive하고 누구나 예상할 수 있는 결론이다.
- 또한 https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002756191과 논문 Sun Woong Kim. (2021). COVID-19 Fear Index and Stock Market. Journal of Convergence for Information Technology, 11(9), 84-93.을 보면
  이미 코로나와 공포지수는 관련이 있다고 보는게 일반적인 사실인데 그래프의 결과는 너무나도 관련이 없는 수치가 나왔다.
  
또한 그래프를 공포지수나 당시 사회 불안을 나타네는 표본들과 비교해 봐도 무언가 사회 현상과 연상지을 만한 부분을 발견하지는 못했으며, 이후 전체 데이터에서 무작위로 천개를 뽑아 직접 데이터를 읽어본 결과
false가 나오더라도 코로나나 백신과 직접 관련된 코멘트가 다분했고, 긍/부정 또한 부정으로 나왔지만, 중립에 가깝거나 아예 관련이 없는 정치 이야기인 경우가 너무 많았다.


## 결론
전처리 방법이 무언가 잘못 되었다는 것을 인식하고 처음부터 데이터 전처리부터 다시하기로, 방법을 바꾸기로 결정했다.

---


## 1️⃣ 다시 한번 개요 및 목표 (Overview & Thesis)

### 🎯 문제 정의 및 프로젝트의 의의  
본 프로젝트는 **COVID-19 팬데믹 기간 동안 온라인 커뮤니티에 남겨진 약 11만 건의 원천 데이터**를 확보하고,  
**합리적이고 단계적인 정제 과정**을 거쳐 최종 분석의 **신뢰성을 극대화**하는 것을 최우선 목표로 삼았습니다.

- **핵심 목표:**  
  데이터 정제 과정을 통해 확보된 **고순도(High-purity) 데이터셋**을 분석하여 새로운 결론을 찾는 것.




---

##  다시 한번 더  데이터 전처리 파이프라인 (노이즈 제거의 합리성)

### 🧠 논리적 동기: 초기 LDA 결과의 문제점  
초기 원천 데이터를 최소 전처리 후 LDA로 분석한 결과,  
**URLs, 감탄사, 정치인 이름 등 가비지 토픽**이 다수 등장 → 고노이즈 데이터로 판정.  
이에 따라 **4단계의 정제 파이프라인**을 설계했습니다.

---

### 🔹 단계 1: 구조적 노이즈 제거 (API Placeholder 및 길이 필터링)

| 가비지 유형 | 적용 기법 | 처리 내용 | 근거 |
|--------------|------------|------------|------|
| [deleted], [No Content] | pandas filtering | 삭제된 게시물 제거 | 의미 없는 구조적 잡음 제거 |
| 짧은 잡담 (lol, ok 등) | length check | 20자 미만 / 5단어 미만 텍스트 제거 | 학습 아웃라이어 제거, 모델 견고성 향상 |

---

### 🔹 단계 2: 언어적 노이즈 제거 (비영어 비율 10% 기준)

| 가비지 유형 | 적용 기법 | 처리 내용 | 근거 |
|--------------|------------|------------|------|
| 한글/타언어 포함 | RegEx filtering | 비영어 비율 10% 초과 시 제거 | 언어적 순도 향상 및 학습 효율 극대화 |

---

### 🔹 단계 3: 형식적 노이즈 제거 (특수문자 및 불용어)

| 가비지 유형 | 적용 기법 | 처리 내용 | 근거 |
|--------------|------------|------------|------|
| 특수문자 / URL | RegEx ratio check | 알파벳 외 문자 40% 초과 제거 | 인코딩 오류 및 스팸 방지 |
| 불용어 | Stopword list | the, a, is, covid, vaccine 등 제거 | 핵심 단어 가중치 향상 및 주제 분리도 개선 |

---

## 4️⃣ 분석의 엄밀성: 최종 주제 관련성 필터링 (Analytical Refinement)

### 🎯 단계 4: 주제 무관 데이터 제거 (Relevance Filtering)
물리적 정제 이후에도 여전히 ‘like, think, time’ 등의 **비주제적 단어**가 남아 있었기 때문에,  
**COVID-19 관련 키워드**를 포함한 데이터만 남기도록 필터링했습니다.


이후, 전처리 전 데이터에서 떠오르는 토픽들이 코로나나 백신과 너무 관련이 없다는 것을 확인하고 키워드를 골라서 그 키워드가 있는 댓글들은 True/ 없는 경우를 false 분류하고 false 댓들들은 삭제하는 전처리를 진행했다.

# 🔍 주제 관련성 키워드 분류 코드 요약

| 구분 | 주요 내용 | 사용 라이브러리 / 함수 | 목적 및 중요성 |
|------|------------|--------------------------|----------------|
| **1단계: 데이터 준비** | `Real_Final.csv` 파일을 로드하고, 결측값을 제거하여 데이터의 무결성을 확보합니다. | `pandas.read_csv()`, `dropna()` | 분석 대상 텍스트 데이터를 정제하고, 초기 데이터 건수를 확인하여 분석의 기준점을 설정합니다. |
| **2단계: 주제 정의 및 확장** | COVID-19 및 백신 관련 핵심 키워드 20여 개를 명시적으로 정의합니다.<br>(예: *vaccine, covid, side effect, mask, booster, mrna* 등) | `Python list` | 논란 분석에 필요한 핵심 주제의 범위를 구체적으로 한정합니다. 정의된 키워드가 포함된 텍스트만 분석 대상으로 삼아 효율성을 높입니다. |
| **3단계: 이진 분류 로직** | 각 텍스트를 소문자화한 후, 정의된 키워드 중 하나라도 포함되어 있는지 여부를 확인하여 **True(관련 있음)** 또는 **False(관련 없음)**으로 분류합니다. | `str.lower()`, `any()` | 데이터셋을 **‘주제 관련 데이터’**와 **‘주제 무관 데이터’**로 구분하여 필터링 기준을 설정합니다. 이후 논란 분석 등 핵심 분석에 집중할 수 있는 기반을 마련합니다. |
| **4단계: 결과 분석 및 저장** | 분류된 결과를 기반으로 관련 데이터의 건수와 비율을 산출하고, 최종적으로 `is_related_topic` 컬럼이 포함된 새로운 CSV(`FINAL_DATA_CLEANED_CLASSIFIED_V2.csv`)로 저장합니다. | `df.to_csv()` | 분류된 데이터의 통계적 분포를 확인하고, **후속 심층 분석(LDA·감성 분석 등)**의 기반 데이터를 확정합니다. |
---

### 💡 보고서 활용 및 분석적 시사점

1. **분석 범위의 명확화**  
   전체 데이터(약 98,277건) 중에서 **핵심 키워드 기반 필터링**을 통해  
   COVID-19 및 백신 관련 텍스트만 선별함으로써,  
   이후 분석(예: 토픽 모델링, 감성 분석)이 **핵심 논의 중심 데이터에 집중**되도록 함.  
   > **보고서 예시 문구:**  
   > “전체 데이터 중 약 **XX%**가 COVID-19/백신 관련 키워드 기반으로 분류되어 후속 분석에 활용되었다.”

2. **키워드 기반 분류의 한계와 보완 (Feat. LDA)**  
   - 키워드 미포함 관련 텍스트 누락, 키워드 포함 비관련 텍스트 포함 등 **오분류 가능성** 존재.  
   - 하지만, **LDA 토픽 모델링 결과를 바탕으로 핵심 키워드를 확장 정의**하여  
     이 단계를 **전략적 필터링 과정**으로 활용할 수 있음.  
     즉, “LDA로 주제 후보를 도출 → 본 코드로 해당 주제를 대표하는 텍스트 선별”의 **2단계 분석 체계**로 설명 가능.


---
## 키워드
KEYWORDS = [
    'vaccine', 'covid', 'coronavirus', 'side effect', 'adverse', 'pfizer', 'moderna',
    'booster', 'jab', 'shot', 'vax', 'myocarditis', 'astrazeneca', 'janssen',
    'symptoms', 'mandate', 'mask', 'masked', 'unvaccinated', 'vaxxed', 'unvaxxed',
    'hospital', 'death', 'long covid', 'long-covid', 'spike protein', 'mrna' ]
 이렇게 잡았다.

 이유는 저번에 'vaccine', 'covid', 'coronavirus', 'side effect' 이런식으로 간단히 잡으니, 코로나 백신과 관련이 있는데도 false로 구분된 경우가 매우 많았기에, 저번 false 데이터(약 7만 건) 중에 1000 건을 무작위로 뽑아 실은True 였던 것들은 직접 라벨링을 한번 더 진행하고, 그 중 빈도가 높게 등장했던 상위 30위를 뽑아 위 합리적인 키워드들을 정했다.

| 키워드 그룹 | 예시 | 포함 이유 |
|--------------|------|------------|
| 백신/의학 | vaccine, covid, pfizer, moderna, booster,pfizer | 백신 관련 직접 논의 |
| 정책/의무 | mandate, mask, masked, unvaccinated,jab | 사회적 의무 논란 |
| 부작용/피해 | side effect, adverse, symptoms, hospital | 의학적 부작용 논의 |

---

### 📊 필터링 결과

| 결과 구분 | 건수 | 비율 | 통찰 |
|------------|------|------|------|
| ✅ True (관련 있음) | 23,939 | 23.6% | 최종 분석용 고순도 데이터 |
| ❌ False (무관함) | 75,338 | 76.4% | 잡담, 뉴스 등 제거 |


false는 모두 지우고 True만 남기여서 FINAL_DATA_FILTERED_TRUE.csv로 저장.

## 검증
이 FINAL_DATA_FILTERED_TRUE.csv 데이터가 과연 정말로 코로나 백신 대이터로써 좋은 데이터인지 검증하기 위해 또 10%(약 2,200개)를 무작위로 뽑아 직접 읽어보면서 코로나와 정말로 관련이 있는지 확인하는 작업을 거쳤다.

결과를 보니 대부분 정말로 코로나와 관련이 있는 데이터였지만, 종종 의료관련업계 사람들이 코로나 백신에 대한 자신의 의견이 없이 여론이 아닌 논문의 링크나 기사의 링크는 보내는 경우가 종종 보였기 때문에 [eX) See the rest of the article by infectious disease expert [Dr. Siouxsie Wiles](https://en.wikipedia.org/wiki/Siouxsie_Wiles) (PhD from Oxford) [here](https://thespinoff.co.nz/society/09-03-2020/the-three-phases-of-covid-19-and-how-we-can-make-it-manageable/).]  추가적으로 FINAL_DATA_FILTERED_TRUE.csv 데이터에서 그런 데이터를 지우는 전처리를 진행했다.
#결과
✅ 원본 데이터 (23939 행) 불러오기 완료.
✅ 클리닝 완료. 총 1010개의 행이 삭제되었습니다.
🎉 클리닝된 데이터 (22929 행)가 'FINAL_DATA_ROWS_DELETED.csv'(으)로 성공적으로 저장되었습니다.

다만, 기사를 인용하면서도 단순 기사 공유가 아닌 경우에도 있을 수 있을 것 같아서 1010개를 직접적으로 확인해 보고 쓸 수 있는 데이터라고 판단한 423개는 다시 추가하여 'FINAL_DATA_ROWS_DELETED_2.csv'(23,352개)로 저장하였다.


> ✔️ **결론:** 데이터의 양보다 질을 선택 — 감성 분석의 초점이 백신 논란의 본질에 집중되도록 보장
>
> 이 정제된 FINAL_DATA_ROWS_DELETED.csv를 사용할 것이다

---


##  5. 단어 빈도 분석 (Word Frequency Analysis)

### 코드 요약

| 단계 | 주요 내용 | 사용 라이브러리 | 목적 |
|------|------------|------------------|------|
| 1️⃣ 데이터 로드 및 전처리 | 불용어 제거, 표제어 추출 | `pandas`, `re`, `nltk` | 분석 정확도 향상 |
| 2️⃣ 단어 빈도 계산 | 전체 문서에서 단어 집계 | `collections.Counter` | 주요 단어 정량 분석 |
| 3️⃣ 상위 50개 키워드 추출 | `most_common()` 사용 | `Counter` | 핵심 관심사 파악 |

> 🔹 **LDA vs. 단어 빈도 비교**
> - LDA: 단어 간 *연관성* 기반 주제 도출  
> - 단어 빈도: 단순 *언급 횟수* 기반 주요 키워드 파악  
>  
> 두 결과를 교차 검증함으로써 분석의 신뢰도를 강화했습니다.


이후 또 토픽 모델링을 통해서 어떤 결과가 나오는지 파악함 
１위는 ｐｅｏｐｌｅ 
２위는 ｃｏｖｉｄ
３ ｖａｃｃｉｎｅ
４ ｇｅｔ
５ ｄｏｎｔ
６ ｍａｓｋ 

였다。 이를통해서 사람들이 확실히 코로나에 대해서 이야기를 하고 있다는 방증을 얻을 수 있었으며 (2위와 3위), 또한 사람들이 일반 대중에 대한 이야기 (1위)를 하고 있다는 점, 부정적인 이야기가 많다는 점 (5위), 그리고 백식을 맞아도 걸린다는 점 (4위) 그리고 추가적으로 마스크에 대한 의견 피력이 많다는 점을 알 수 있었다 (6위)


이후 전체 데이터에서 10% (약 2100개)를 임의로 때어서 직접 라벨링을 진행하였다.

저번에 라벨링을 이진으로 (긍/부정) 으로 분류하다가 느낌점이 중립의 비율이 꽤나 높다는 인상을 받았기 때문에 이진 분류와 삼분류를 동시에 진행했다.

그 비율은 아래와 같다.


 [이진 분류 결과 비율]
sentiment_binary
부정    81.21
긍정    18.79

 [삼분류 결과 비율]
sentiment_three
부정    63.51
긍정    18.79
중립    17.70

이후 Koelectra 모델을 이용해 머신러닝하여 각각의 validation accuracy와 train rose 를 구하고 이진과 삼진으로 나누어 row data에 적용하여 모델로 라벨링을 진행했다.

결과는 아래와 같다.


Ⅰ. 데이터 개요 및 탐색적 분석 (EDA)목표: 댓글 데이터에서 자주 등장하는 단어를 파악하고, 주요 키워드의 트렌드를 시계열적으로 분석하여 데이터의 초기 특성을 이해합니다.1. 전처리 및 주요 단어 추출전처리: 텍스트를 소문자 변환, URL 제거, 비알파벳 문자 제거 후 불용어(Stopwords) 및 3글자 미만 단어를 제거했습니다.결과물:top_words_frequency.png: 상위 20개 단어 빈도 막대 그래프 (예: 'covid', 'vaccine', 'mask' 등이 상위권 차지).2. 시계열 단어 빈도 분석방법론: 상위 5개 단어를 선정하여 월별로 상대적 빈도 (1,000단어당 등장 횟수)를 계산하고, 시간에 따른 관심도 변화를 추적했습니다.결과물:word_frequency_over_time.png: 상위 5개 단어의 월별 상대적 빈도 변화 꺾은선 그래프.monthly_word_frequency_ts.csv: 월별 빈도수 데이터 (보고서 자료).Ⅱ. 딥러닝 학습 데이터 준비 및 라벨링 (Human Annotation)목표: 딥러닝 모델(KoElectra) 학습을 위한 고품질의 수동 라벨링 데이터를 준비합니다.1. 10% 무작위 샘플링 및 인코딩 문제 해결샘플링: 원본 데이터(약 22,939개) 중 **10%**인 2,294개의 댓글을 무작위 샘플링하여 학습 데이터셋으로 사용했습니다.핵심 수정: 데이터 로드 시 발생한 깨진 문자(Mojibake) 문제를 해결하기 위해 pd.read_csv 함수에 encoding='utf-8' 또는 encoding='cp949' 옵션을 명시적으로 지정했습니다.라벨링 파일: **시간 정보(created_at)**를 포함하여 추후 감성 시계열 분석에 활용할 수 있도록 최종 준비 파일을 생성했습니다.사용자 작업: 이 샘플 파일에 대해 수동 라벨링을 진행하여 다음과 같은 두 가지 파일을 생성했습니다.BERT_labeled_binary.csv (긍정/부정)BERT_labeled_three.csv (긍정/부정/중립)Ⅲ. KoElectra 모델 학습 및 평가 (Deep Learning)목표: KoElectra 모델을 사용하여 Binary 및 Three-Class 감성 분류를 수행하고, 모델의 Valid Accuracy 및 원본 데이터 예측 결과를 도출합니다.1. 학습 환경 및 모델 설정항목설정 내용비고모델KoElectra (monologg/koelectra-base-v3-discriminator)한국어 자연어 처리 모델평가 지표Valid Accuracy모델의 일반화 성능 측정데이터 분리라벨링된 샘플 중 **90%**는 학습(Training), **10%**는 **검증(Validation)**에 사용.신뢰성 있는 Valid Accuracy 확보 목적버전 호환성TrainingArguments의 evaluation_strategy, save_strategy 오류 해결사용자 환경에 맞춰 eval_steps=100, save_steps=100 (스텝 기반 저장) 방식으로 코드를 수정하여 실행 가능하게 함.2. 두 가지 독립적인 학습 작업작업데이터 파일목표 클래스라벨 매핑출력 파일 (예측)BinaryBERT_labeled_binary.csv2개 (부정, 긍정)부정: 0, 긍정: 1predicted_binary.csv (원본 전체 예측)Three-ClassBERT_labeled_three.csv3개 (부정, 중립, 긍정)부정: 0, 중립: 1, 긍정: 2predicted_three.csv (원본 전체 예측)3. 최종 기대 결과학습 코드를 성공적으로 실행하면 다음과 같은 세 가지 주요 결과를 얻게 됩니다.Valid Accuracy 값 (Binary 및 Three-Class 각각)predicted_binary.csv: 원본 데이터 전체에 긍정/부정 감성 라벨이 추가된 파일.predicted_three.csv: 원본 데이터 전체에 긍정/부정/중립 감성 라벨이 추가된 파일.


이 모두 '0'으로 나오는 현상은 딥러닝 감성 분석 모델에서 흔히 발생하는 "예측 붕괴(Prediction Collapse)" 또는 "과도한 편향 학습(Bias Learning)" 문제입니다.이 현상은 모델이 학습 데이터의 패턴을 충분히 익히지 못하고, 손실(Loss)을 가장 빠르게 줄일 수 있는 방법, 즉 가장 많은 비율을 차지하는 클래스(Majority Class)만 예측하도록 편향되기 때문에 발생합니다.1. 🛑 문제의 근본 원인 분석우선, 수동 라벨링하신 데이터의 실제 클래스 분포를 분석한 결과는 다음과 같습니다.분류라벨개수비율매핑된 숫자Binary부정1,86381.2%0긍정43118.8%1Three-Class부정1,45763.5%0긍정43118.8%2중립40617.7%11) 클래스 불균형 (Class Imbalance)Binary: '부정'이 **81.2%**로 압도적입니다. 모델이 모든 것을 '부정'(0)으로 예측해도 81.2%의 정확도를 달성할 수 있습니다.Three-Class: '부정'이 **63.5%**로 여전히 다수를 차지합니다.모델이 2 Epochs라는 짧은 학습 시간 동안 이 심각한 불균형을 극복하고 소수 클래스(긍정, 중립)의 특징을 학습하기에는 역부족이었습니다. 따라서 모델은 **가장 안전한 선택인 '0'(부정)**만을 예측하도록 편향된 것입니다.2) 짧은 학습 시간 (Epochs=2)이전에 안내해 드린 대로, 2 에포크는 Fine-tuning 초기 단계에 불과하며, 모델이 불균형을 극복하고 각 클래스의 미묘한 차이를 배우기에는 시간이 너무 짧았습니다.2. ✅ 해결책: 에포크 수 증가 및 가중치 조정이 문제를 해결하기 위해 가장 쉽고 효과적인 방법은 학습 시간을 늘리는 것입니다.1) 에포크 수 증가 (1순위 해결책)학습을 더 오래 진행하여 모델이 소수 클래스(긍정/중립)의 특징을 더 깊이 학습하도록 유도해야 합니다.수정: 에포크 수를 2에서 5 또는 10 정도로 늘려 재시도해 보세요.2) 클래스 가중치 적용 (심화 해결책)클래스 불균형이 너무 심하므로, 손실 함수(Loss Function)에 **클래스 가중치(Class Weights)**를 적용하여 소수 클래스(긍정/중립)의 오분류에 더 큰 패널티를 부여해야 합니다.방법: '부정'의 가중치를 낮추고, '긍정'과 '중립'의 가중치를 높여 모델이 소수 클래스를 놓치지 않도록 강제합니다. (이는 PyTorch의 CrossEntropyLoss 함수에 weight 인자를 전달하여 구현할 수 있습니다.)

---
===== BERT_labeled_binary.csv 모델 학습 시작 (클래스 수: 2) =====
Some weights of ElectraForSequenceClassification were not initialized from the model checkpoint at monologg/koelectra-base-v3-discriminator and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Epoch 1: 100%|██████████| 115/115 [00:15<00:00,  7.45it/s]
Epoch 2:   0%|          | 0/115 [00:00<?, ?it/s]Epoch 1 | Train Loss: 0.4909
Epoch 2: 100%|██████████| 115/115 [00:14<00:00,  7.97it/s]
Epoch 2 | Train Loss: 0.4813
✅ Validation Accuracy: 0.8126
Predicting FINAL dataset: 100%|██████████| 717/717 [01:13<00:00,  9.82it/s]
💾 예측 결과 저장 완료: predicted_binary_2.csv
========================================


===== BERT_labeled_three.csv 모델 학습 시작 (클래스 수: 3) =====
Some weights of ElectraForSequenceClassification were not initialized from the model checkpoint at monologg/koelectra-base-v3-discriminator and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Epoch 1: 100%|██████████| 115/115 [00:13<00:00,  8.37it/s]
Epoch 1 | Train Loss: 0.9314
Epoch 2: 100%|██████████| 115/115 [00:13<00:00,  8.45it/s]
Epoch 2 | Train Loss: 0.9033
✅ Validation Accuracy: 0.6362
Predicting FINAL dataset: 100%|██████████| 717/717 [01:07<00:00, 10.64it/s]
💾 예측 결과 저장 완료: predicted_three_2.csv
========================================

🎯 최종 결과
Binary Validation Accuracy : 0.8126
Three-class Validation Accuracy : 0.6362


---


이후 이진과 삼진 데이터의 긍/부정/중립 비율을 시계열에 따라 정리했다.
결과는 아래와 같다.

---

이후 긍정 데이터만 토픽 모델링 진행

이후 부정 데이터만 토픽 모델링 진행.

결과는 아래와 같다.















