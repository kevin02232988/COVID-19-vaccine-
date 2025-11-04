![header](https://capsule-render.vercel.app/api?type=soft&color=auto&height=300&section=header&text=vaccine%20Review💉&fontSize=90)<br/>
# COVID Vaccine Controversy Analysis by BERT 🦠
Multilingual BERT를 활용한 코로나 백신 여론 분석 프로젝트

[![PyTorch](https://img.shields.io/badge/PyTorch-E34F26?style=flat-square&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21C?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white)](https://www.python.org/)

---

## 1. 개요 및 목표 (Overview & Thesis)

### 1-1. 문제 정의 및 프로젝트의 의의
본 프로젝트는 COVID-19 팬데믹 기간에서 벗어난 지금 이 시전에서  그동안 온라인 커뮤니티에 뎃글과 리뷰들을 모아서 정제해보고 그 정제된 데이터를 통해 무언가 결론을 찾아보는 것을 목표로 합니다. 

- **핵심 목표**: 약 10만건의 데이터를 분석하여 사회적 의무(Mandate) 관련 논란이 과학적 부작용보다 여론 확산에 미친 영향이 더 컸음을 입증
- **성공 지표**: 1만건의 수동 라벨링 데이터를 기반으로 F1 Score 0.6438 달성 → 모델 신뢰성 검증


- **핵심 목표**: 98,277건의 데이터를 분석하여 사회적 의무(Mandate) 관련 논란이 과학적 부작용보다 여론 확산에 미친 영향이 더 컸음을 입증
- **성공 지표**: 9,827건의 수동 라벨링 데이터를 기반으로 F1 Score 0.6438 달성 → 모델 신뢰성 검증

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
- **최종 규모**: 약11만 건의 영문및 한글 게시글 및 댓글

- **이후, 한국어 데이터는 사용하지 않기로 결정해야, 영어 데이터만 남도록 전처리 실시함.



##  데이터 통합 및 전처리 과정 (실제 적용 **4단계**)

저희는 수많은 소스를 통합하고 양질의 데이터를 확보하기 위해 다음 **4단계의 실제 정제 파이프라인**을 적용했습니다.

---

###  구조적 노이즈 (API Placeholder) 제거

| 가비지 유형 | 문제 원인 | 적용 기법 | 처리 내용 |
|:--|:--|:--|:--|
| **API 삭제 항목** | $\text{Reddit}$ $\text{API}$에서 삭제된 게시글/댓글이 `[No Content]` 또는 `[deleted]`로 표시됨 | $\text{Pandas}$ $\text{Filtering}$ | 해당 **플레이스홀더 문자열이 포함된 모든 행**을 제거하여 데이터의 유효성을 확보함 |
| **짧은 잡담** | 텍스트가 매우 짧은 댓글 (예: `"lol"`, `"ok"`)은 감정 분석에 노이즈로 작용 | $\text{Length}$ $\text{Check}$ | **20자 미만 또는 5단어 미만**인 텍스트를 제거하여 분석 순도 강화 |

---

###  언어적 / 국가별 노이즈 필터링 (한국어 Naver 데이터 분리)

| 가비지 유형 | 문제 원인 | 적용 기법 | 처리 내용 |
|:--|:--|:--|:--|
| **Non-English** | 초기 크롤링 시도 ($\text{Naver}$)에서 유입된 한국어 Q&A 데이터 | $\text{RegEx}$ $\text{Based}$ $\text{Filtering}$ | 텍스트 내 **한글 문자 비율이 10%를 초과**하는 행을 제거하여 $\text{BERT}$ 학습의 **영문 순도**를 극대화함 |

---

###  형식적 노이즈 (Special Char) 제거

| 가비지 유형 | 문제 원인 | 적용 기법 | 처리 내용 |
|:--|:--|:--|:--|
| **과도한 특수문자** | URL, 이모티콘, 반복 기호 등 | $\text{RegEx}$ $\text{Ratio}$ $\text{Check}$ | **알파벳 외 문자 비율이 40% 초과**하는 행을 제거하여 토큰화 전 데이터의 노이즈 제거 |
| **불용어 제거** | `'the'`, `'a'`, `'is'` 등 의미 없는 단어와 `'covid'`, `'vaccine'` 등 빈번한 주제 단어 | $\text{Stopword}$ $\text{List}$ | **핵심 논란 키워드** (`mask`, `mandate`)를 정확히 추출할 수 있도록 토픽 모델링 효율 증대 |

---
이후 텍스트 마이닝 토픽 모델링을 진행함

텍스트를 전처리하는 함수: 소문자화, 불필요한 문자 제거, 토큰화, 불용어 제거, 표제어 추출."""
토픽 모델링 코드 요약 (보고서용)
구분	주요 내용	사용 라이브러리 / 함수	목적
1단계: 데이터 로드	Real_Final.csv 파일을 Pandas DataFrame으로 불러옵니다.	pandas.read_csv()	분석할 데이터 준비
2단계: 데이터 전처리 (핵심)	텍스트 데이터를 LDA 분석에 적합하도록 정제합니다.	nltk (토큰화, 표제어 추출), re (URL 제거)	노이즈 제거 및 단어 정규화
➤ 세부 과정 1	URL, 숫자, 특수문자 제거	re.sub()	분석 방해 요소 제거
➤ 세부 과정 2	토큰화(단어 분리) 및 불용어(the, is 등) 제거	nltk.word_tokenize(), nltk.corpus.stopwords	의미 없는 단어 제거
➤ 세부 과정 3	표제어 추출(Lemmatization)	nltk.stem.WordNetLemmatizer	단어를 원형으로 통일 (running → run)
3단계: 문서-단어 행렬(DTM) 생성	전처리된 단어를 BoW(Bag-of-Words) 형태로 변환하여 모델 입력 형태로 만듭니다.	gensim.corpora.Dictionary, doc2bow()	텍스트 데이터를 수치화
➤ 세부 과정 1	사전(Dictionary) 생성	Dictionary()	고유 단어에 ID 부여
➤ 세부 과정 2	필터링: 너무 낮거나 높은 빈도의 단어 제거	filter_extremes(no_below=5, no_above=0.5)	모델 성능 저해 단어 제거
➤ 세부 과정 3	코퍼스(Corpus) 생성	doc2bow()	문서별 단어 빈도 계산
4단계: LDA 모델 학습	전처리된 코퍼스를 기반으로 10개의 잠재 토픽을 학습	gensim.models.LdaModel	토픽 추출 및 단어-토픽 분포 학습
➤ 주요 파라미터	num_topics=10, passes=20, random_state=42	토픽 개수 및 반복 횟수 지정	
5단계: 결과 해석	학습된 10개 토픽 각각의 상위 10개 키워드를 출력하여 의미를 해석	lda_model.print_topics()	분석 결과 도출
📊 LDA (Latent Dirichlet Allocation) 모델 설명

LDA는 잠재된 주제를 발견하는 비지도 학습(Unsupervised Learning) 기반의 토픽 모델링 알고리즘입니다.

모든 문서는 몇 가지 주제(Topic) 의 혼합으로 구성되어 있고, 각 주제는 몇 가지 단어(Word) 의 혼합으로 구성되어 있다고 가정합니다.

즉, 문서 전체를 읽지 않고도 텍스트의 주요 논의 흐름과 숨겨진 의미 구조를 파악할 수 있습니다.
