# 데이터 전처리 고도화 실행 계획 (MoSCoW TODO Checklist)

## 목적
- `data/raw/train.csv`, `data/raw/test.csv`를 `data.py`에서 **예외 없이** 처리
- train↔test의 **스키마/표현 불일치**를 줄여 OOV(학습에서 못 본 값)로 인한 정보 손실 최소화
- `config/preprocess/default.yaml`를 통해 전처리 동작을 **Hydra 기반으로 제어**(옵션 on/off, 파서 선택, 하이퍼파라미터)

## 성공 기준(Definition of Done)
- `preprocess_data(..., is_train=True)` / `preprocess_data(..., is_train=False)`가 동일 실행 흐름에서 오류 없이 동작
- inference 시 전처리 결과 컬럼이 train과 동일한 순서/스키마로 생성(타깃 제외)
- 주요 품질 이슈 3종이 재발하지 않음
  - 전공 스키마 불일치(`major1_1`, `major1_2`)
  - 컬럼별 delimiter 불일치로 인한 토큰 깨짐(`onedayclass_topic`, `expected_domain`, `certificate_*`)
  - `completed_semester` 이상치(예: `2020.02`, `20241`)

## 데이터 품질 이슈(근거 스냅샷)
- `major1_1` nunique: train=12 vs test=289 (대분류 vs 학과명 수준)
- `major1_2` nunique: train=11 vs test=154
- `expected_domain` nunique: train=170 vs test=19 (문장 내 쉼표 다수 + 코드형)
- `onedayclass_topic`: 항목 내부 괄호에 쉼표 존재(`Matplotlib, Seaborn` 등) → 단순 `split(",")` 실패
- `expected_domain`: 항목 내부에 쉼표 다수, 실제 구분은 `A.`/`B.` 형태 코드 → 단순 `split(",")` 실패

## 작업 대상(수정 예정 파일)
- `data.py` (전처리 로직/피처 생성/파서)
- `config/preprocess/default.yaml` (Hydra 전처리 설정)

## 컬럼별 파서/정규화 매트릭스(초안)

### 파서 타입 정의(제안)
- `code4_prefix`: `0001:` 같은 **4자리 코드**만 추출(설명 텍스트 무시)
- `alpha_dot`: `A.` 같은 **letter-dot 코드**만 추출(내부 쉼표/문장 무시)
- `comma_outside_parens`: 괄호 내부(`(...)`) 쉼표는 보호하고 **괄호 밖 쉼표만** 구분자로 사용
- `certificate_normalize`: 자격증 표기 흔들림(띄어쓰기/괄호/등급/영문 대소문자)을 정규화 후 토큰화(또는 슬롯화로 연결)

### 컬럼→파서 매핑(제안)
- `previous_class_3~8` → `code4_prefix`
- `desired_job`, `desired_job_except_data`, `expected_domain` → `alpha_dot`
- `onedayclass_topic` → `comma_outside_parens`
- `certificate_acquisition`, `desired_certificate` → `certificate_normalize` (또는 슬롯 파서로 확장)

## 전공 대분류 매핑 규칙(v1) 초안
- IT: `컴퓨터`, `소프트웨어`, `정보`, `데이터`, `AI`, `인공지능`, `전산`, `전자`, `통신`, `빅데이터`
- 경영학: `경영`, `회계`, `경영정보`, `재무`, `마케팅`
- 경제통상학: `경제`, `통상`, `무역`
- 자연과학: `수학`, `통계`, `물리`, `화학`, `생명`, `지구`, `환경`
- 사회과학: `심리`, `사회`, `정치`, `행정`, `국제`, `커뮤니케이션`
- 인문학: `국문`, `영문`, `문헌`, `사학`, `철학`, `언어`, `역사`
- 의약학/교육학/법학/예체능: 해당 키워드 매칭 우선(충돌 시 우선순위 규칙 필요)
- 기타: 매핑 실패 시 `OTHER`

> 구현 시 권장: (1) train의 대분류 값은 그대로 통과, (2) test의 학과명은 suffix(`학과/학부/전공`) 제거 후 키워드 매칭, (3) 다중 매칭 시 우선순위/점수 기반 선택 + `major_map_ambiguous` 플래그.

---

## Must (M) — 이번 개정에서 반드시

- [x] (M1) train↔test 스키마 점검 결과를 문서로 고정
  - [x] 컬럼 목록/타입/결측률/카디널리티(특히 major, multi-select)를 본 문서에 반영
  - [x] `major1_1`, `major1_2` 상위 빈도 값(상위 20) train/test 각각 기록
  - [x] 스키마 스냅샷
    - train cols=46(타깃 포함), test cols=45(타깃 없음)
    - `major1_1` top-20 (train): IT(253), 경영학(178), 자연과학(97), 사회과학(57), 인문학(48), 경제통상학(36), 의약학(29), <NA>(20), 예체능(15), 교육학(10), 법학(3), 기타(2)
    - `major1_1` top-20 (test): <NA>(73), 경영학과(47), 통계학과(46), 응용통계학과(30), 경영학부(26), 컴퓨터공학과(23), 정보통계학과(19), 수학과(18), 산업공학과(15), 데이터사이언스학과(14), 소프트웨어학과(13), 산업경영공학과(12), 인공지능학과(11), 정보통신공학과(11), 경제학과(10), 컴퓨터공학부(10), 심리학과(8), 문헌정보학과(8), 컴퓨터학부(8), 산업시스템공학과(7)
    - `major1_2` top-20 (train): <NA>(439), IT(139), 경영학(66), 자연과학(53), 사회과학(19), 경제통상학(18), 기타(4), 예체능(4), 의약학(3), 인문학(2), 교육학(1)
    - `major1_2` top-20 (test): 없음(336), <NA>(183), 통계학과(23), 경영학과(19), 컴퓨터공학과(16), 응용통계학과(13), 경제학과(10), 경영학부(8), 데이터사이언스학과(7), 사회학과(6), 산업공학과(6), ...

- [x] (M2) `major1_1`, `major1_2` 스키마 정합(대분류 매핑) 설계 및 구현
  - [x] 규칙 기반 키워드 매핑(예: IT/경영/경제통상/자연과학/사회과학/인문/의약/교육/법학/예체능/기타)
  - [x] 파생 컬럼(`major1_1_coarse`, `major1_2_coarse`) 추가(원본 유지)
  - [x] 매핑 실패/모호 케이스는 `OTHER`로 폴백 + `major_map_failed` 플래그 생성
  - [x] 매핑 기능 on/off, ruleset(또는 키워드 사전)을 `default.yaml`에서 제어 가능하게 설계

- [x] (M3) 멀티선택/코드형 컬럼의 파싱 전략 분리(쉼표 split 오류 제거)
  - [x] `previous_class_3~8`: `0001:` 같은 4자리 코드만 안정 추출(텍스트 설명은 무시)
  - [x] `desired_job`, `desired_job_except_data`, `expected_domain`: `A.`/`B.` 형태 letter-dot 코드만 추출
  - [x] `onedayclass_topic`: 괄호 내부 쉼표는 보호하고 항목 단위로 split(예: `,` 중 괄호 밖만 구분자로 사용)
  - [x] `certificate_acquisition`, `desired_certificate`: 괄호/등급/띄어쓰기 변형을 흡수하는 정규화(예: `컴퓨터 활용능력(1,2급)` → `컴퓨터활용능력`)
  - [x] 파서 선택을 `default.yaml`에서 컬럼별/패턴별로 지정 가능하도록 설계(예: `multilabel_parser_rules`)

- [x] (M4) `completed_semester` 이상치 클린업 + 플래그
  - [x] 정수 범위를 벗어나거나 소수점이면 `NaN` 처리(or clip) + `completed_semester_is_outlier` 생성
  - [x] 정제 후 기본 통계(최솟값/최댓값/이상치 개수)를 본 문서에 기록
  - [x] 이상치 예시(학습): `2020.02`, `20241` 등 총 3건

- [x] (M5) `config/preprocess/default.yaml` 개정(신규 옵션 추가 + 하위호환 유지)
  - [x] 신규 옵션을 `PreprocessConfig`와 1:1 매칭되게 추가
  - [x] 기존 옵션 키(`enable_multilabel_topk`, `text_mode` 등) 유지
  - [x] 기본값은 “현재 동작 최대 유지 + 필수 품질 이슈만 해결”에 맞추기

- [x] (M6) E2E 검증 절차 정의(로컬 실행 기준)
  - [x] `python train.py model=xgboost do_inference=True need_valid=False`로 전처리→추론까지 완료 확인 (lightgbm 미설치로 xgboost로 대체)
  - [x] train/test 전처리 결과 컬럼 수/순서 동일(타겟 제외) 확인
  - [x] 파서별 스모크 테스트(major 매핑/expected_domain/onedayclass_topic/previous_class) 수행

---

## Should (S) — 하면 점수/안정성이 유의미하게 좋아질 것

- [x] (S1) `NaN(미응답)` vs `NONE(없음)` 분리 피처 추가
  - [x] 전 컬럼 공통 정책: `is_missing_*`, `is_none_*`, `is_answered_*` 정의
  - [x] 전체/그룹별 `num_none_*`, `num_answered_*` 집계 피처 추가(응답 성실도 proxy)

- [x] (S2) 자격증 컬럼 “슬롯 기반” 피처(저노이즈, 표현 흔들림에 강함)
  - [x] 슬롯 후보(ADsP/SQLD/정보처리기사/빅데이터분석기사/컴활/태블로 등) 및 동의어 정의
  - [x] `acq_has_*`, `des_has_*`, `acq_cnt`, `des_cnt`, `overlap_cnt`, `need_cnt` 생성
  - [x] 슬롯 목록/동의어를 `default.yaml`에서 관리 가능하도록 설계

- [x] (S3) `previous_class_*` 고도화(ROI 높음)
  - [x] 코드 기반 집계(`prev_taken_cnt`, `prev_unique_code_cnt`) + 주요 코드 플래그(상위 N개) 옵션화
  - [x] 과목명 키워드(python/sql/pandas/ml/stat 등) 카운트 추가(룰 기반, `default.yaml`에서 관리)

- [x] (S4) 멀티라벨 컬럼용 저차원 임베딩(TF‑IDF→SVD) 확장
  - [x] 대상: `interested_company`, `certificate_*` (컬럼별 파서/정규화 후)
  - [x] 차원 수/대상 컬럼을 `default.yaml`에서 제어 가능하게 설계

- [x] (S5) 결측 패턴/응답 패턴 피처 확장
  - [x] `num_answered_total`, `num_missing_total` 및 그룹별 집계 추가
  - [x] KMeans `K` 다중 해상도(예: 3/5/7) 옵션화(`cluster_ks`)

---

## Could (C) — 시간/리스크 대비 여유 있을 때

- [x] (C1) Soft clustering(GMM) 멤버십 확률 피처
- [x] (C2) transductive fit(규정 허용 시): TF‑IDF/SVD/클러스터를 train+test로 fit하여 OOV 완화
- [x] (C3) 공변량 쉬프트 피처: `p(test|x)` 도메인 분류기 기반 유사도 스코어
  - [x] `config/preprocess/default.yaml`에서 `enable_domain_adaptation_feature: true` + `transductive_fit_csv` 지정 시 `p_test` 피처 생성
- [x] (C4) F1 특화: 군집별 threshold(모델/추론 로직까지 확장 필요)
  - [x] `config/config.yaml`의 `cluster_threshold.enabled: true`로 활성화(need_valid=true 필요, 기본 fallback=global threshold)

---

## Won't (W) — 이번 범위에서는 제외(추후 재검토)

- [ ] (W1) 외부 온톨로지/학과 DB 연동(대규모 매핑 자동화)
- [ ] (W2) 라벨 노이즈 탐지/정정(Confident Learning 등)
- [ ] (W3) 대형 LLM 기반 텍스트 이해 피처(비용/재현성 이슈)

---

## `config/preprocess/default.yaml` 개정 방향(키 스케치)

> 실제 적용 시에는 `PreprocessConfig`(data.py) 필드와 1:1로 맞추되, 아래처럼 “파서 규칙/매핑 사전”은 dict/list 형태로 받도록 설계하는 것이 작업/실험 효율이 좋습니다.

```yaml
# 기존 키(유지)
very_sparse_threshold: 0.95
rare_threshold: 5
enable_multilabel_topk: true
multilabel_top_k: 15
multilabel_min_df: 2
text_mode: T2
text_svd_components: 64
enable_clustering: true
cluster_k: 5
cluster_random_state: 42

# 신규(예시) — data.py에서 사용할 필드로 구체화 필요
enable_major_mapping: true
major_mapping_ruleset: v1

# 컬럼/패턴별 파서 지정(우선순위: 위→아래)
multilabel_parser_rules:
  - match: "^previous_class_"
    parser: code4_prefix
  - match: "^(desired_job|desired_job_except_data|expected_domain)$"
    parser: alpha_dot
  - match: "^onedayclass_topic$"
    parser: comma_outside_parens
  - match: "^(certificate_acquisition|desired_certificate)$"
    parser: certificate_normalize

enable_none_features: true
none_strings:
  - "해당없음"
  - "해당 없음"
  - "없음"
  - "없습니다"
  - "무"

completed_semester_valid_min: 0
completed_semester_valid_max: 20
completed_semester_invalid_to_nan: true
```
