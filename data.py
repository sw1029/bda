"""
데이터 전처리용.

- 고정 drop 규칙 + 결측 패턴 피처
- numeric: median impute + is_missing
- categorical: freq/log_freq/is_rare (+ optional top-k one-hot)
- 멀티라벨(쉼표 나열): 집계형 + optional top-k token binarization
- 텍스트: T0/T1/T2 중 선택 (기본 T1)
- 결측 패턴 기반 군집(KMeans) 피처(옵션)
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional

import numpy as np
import pandas as pd


_MISSING_STRINGS = {
    "__missing__",
    "nan",
    "none",
    "null",
}

_EMPTY_SELECTION_TOKENS = {
    "해당없음",
    "해당 없음",
    "없음",
    "없습니다",
    "무",
}

_EMPTY_SELECTION_TOKENS_LOWER = {t.lower() for t in _EMPTY_SELECTION_TOKENS}


def _snake_case_name(name: Any) -> str:
    raw = str(name).strip().lower()
    raw = re.sub(r"[^0-9a-zA-Z가-힣]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or "col"


def _ensure_unique_prefixes(columns: list[str]) -> dict[str, str]:
    prefixes: dict[str, str] = {}
    used: set[str] = set()
    for col in columns:
        base = _snake_case_name(col)
        candidate = base
        i = 2
        while candidate in used:
            candidate = f"{base}_{i}"
            i += 1
        prefixes[col] = candidate
        used.add(candidate)
    return prefixes


@dataclass(frozen=True)
class PreprocessConfig:
    very_sparse_threshold: float = 0.95
    rare_threshold: int = 5

    enable_categorical_onehot: bool = True
    categorical_onehot_max_unique: int = 8
    categorical_onehot_top_k: int = 5

    enable_multilabel_topk: bool = True
    multilabel_top_k: int = 15
    multilabel_min_df: int = 2

    text_mode: str = "T1"  # "T0"|"T1"|"T2"
    text_svd_components: int = 64

    enable_clustering: bool = True
    cluster_k: int = 5
    cluster_random_state: int = 42

    drop_constant_features: bool = True


class TabularPreprocessor:
    def __init__(
        self,
        *,
        id_label: str = "ID",
        target_label: str = "completed",
        config: Optional[PreprocessConfig] = None,
    ) -> None:
        self.id_label = id_label
        self.target_label = target_label
        self.config = config or PreprocessConfig()

        self._is_fitted = False

        self._raw_feature_cols: list[str] = []
        self._prefix: dict[str, str] = {}

        self.drop_all_missing_cols: list[str] = []
        self.very_sparse_cols: list[str] = []

        self.prev_cols: list[str] = []
        self.class_cols: list[str] = []
        self.answer_groups: dict[str, list[str]] = {}

        self.numeric_cols: list[str] = []
        self.numeric_medians: dict[str, float] = {}

        self.categorical_cols: list[str] = []
        self.cat_value_counts: dict[str, dict[str, int]] = {}
        self.cat_top_values: dict[str, list[str]] = {}

        self.multilabel_cols: list[str] = []
        self.multilabel_top_tokens: dict[str, list[str]] = {}

        self.text_cols: list[str] = []
        self._text_models: dict[str, tuple[Any, Any]] = {}

        self._cluster_cols: list[str] = []
        self._kmeans: Any = None

        self.train_columns_: list[str] = []
        self.inference_columns_: list[str] = []
        self._n_train_rows: int = 0

    def _is_missing(self, series: pd.Series) -> pd.Series:
        if series.dtype == object:
            s = series.astype("string")
            lowered = s.str.strip().str.lower()
            return series.isna() | lowered.eq("") | lowered.isin(_MISSING_STRINGS)
        return series.isna()

    def _normalize_categorical(self, series: pd.Series) -> pd.Series:
        missing = self._is_missing(series)
        s = series.astype("string").str.strip()
        s = s.where(~missing, "__MISSING__")
        return s.fillna("__MISSING__")

    def _normalize_token(self, token: str) -> str:
        token = str(token).strip()
        if not token:
            return ""
        lowered = token.lower()
        if lowered in _MISSING_STRINGS:
            return ""
        if lowered in _EMPTY_SELECTION_TOKENS_LOWER:
            return ""

        m = re.match(r"^(\d{4})\s*[:：]", token)
        if m:
            return m.group(1)

        m = re.match(r"^([A-Za-z])\s*\.", token)
        if m:
            return m.group(1).upper()

        return token

    def _split_multilabel(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        missing = self._is_missing(series)
        s = series.astype("string").fillna("").astype(str)

        def to_tokens(value: str) -> list[str]:
            raw_tokens = [t.strip() for t in str(value).split(",")]
            normalized = [self._normalize_token(t) for t in raw_tokens]
            return [t for t in normalized if t]

        tokens = s.where(~missing, "").apply(to_tokens)

        def unique_tokens(items: list[str]) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for item in items:
                if item in seen:
                    continue
                seen.add(item)
                out.append(item)
            return out

        unique = tokens.apply(unique_tokens)
        return tokens, unique

    def _infer_text_cols(self, df: pd.DataFrame, candidate_cols: list[str]) -> list[str]:
        text_cols: list[str] = []
        for col in candidate_cols:
            s = df[col]
            missing = self._is_missing(s)
            non_missing = s.where(~missing, np.nan).dropna()
            if non_missing.empty:
                continue
            nunique = int(non_missing.nunique(dropna=True))
            avg_len = float(non_missing.astype(str).str.len().mean())
            if nunique >= 200 and avg_len >= 15:
                text_cols.append(col)
        if "incumbents_lecture_scale_reason" in candidate_cols and "incumbents_lecture_scale_reason" not in text_cols:
            text_cols.append("incumbents_lecture_scale_reason")
        return text_cols

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        if self.id_label not in df.columns:
            raise ValueError(f"id_label({self.id_label}) 컬럼이 없습니다.")
        if self.target_label not in df.columns:
            raise ValueError(f"target_label({self.target_label}) 컬럼이 없습니다.")

        self._n_train_rows = int(len(df))
        base_cols = [c for c in df.columns if c not in {self.id_label, self.target_label}]
        self._raw_feature_cols = list(base_cols)
        self._prefix = _ensure_unique_prefixes(base_cols)

        missing_rate = {c: float(self._is_missing(df[c]).mean()) for c in base_cols}
        self.drop_all_missing_cols = [c for c, r in missing_rate.items() if r >= 1.0]
        self.very_sparse_cols = [
            c for c, r in missing_rate.items() if (r >= self.config.very_sparse_threshold and r < 1.0)
        ]

        self.prev_cols = [c for c in base_cols if c.startswith("previous_class_")]
        self.class_cols = [c for c in ["class1", "class2", "class3", "class4"] if c in base_cols]

        self.answer_groups = {
            "prev": self.prev_cols,
            "class": self.class_cols,
            "job": [c for c in ["job", "desired_job", "desired_job_except_data", "desired_career_path"] if c in base_cols],
            "company": [c for c in ["interested_company", "incumbents_company_level"] if c in base_cols],
            "cert": [c for c in ["certificate_acquisition", "desired_certificate"] if c in base_cols],
            "major": [
                c
                for c in ["major type", "major1_1", "major1_2", "major_field", "major_data"]
                if c in base_cols
            ],
        }

        # ---- 타입 추론 ----
        force_categorical_numeric = {"school1", "class1", "class2", "class3", "class4"}
        numeric_candidates = [
            c
            for c in base_cols
            if c not in self.drop_all_missing_cols
            and c not in self.very_sparse_cols
            and c not in force_categorical_numeric
            and df[c].dtype.kind in {"i", "u", "f", "b"}
        ]
        self.numeric_cols = list(numeric_candidates)

        # numeric median
        self.numeric_medians = {}
        for col in self.numeric_cols:
            missing = self._is_missing(df[col])
            values = pd.to_numeric(df[col].where(~missing, np.nan), errors="coerce")
            median = float(values.median()) if np.isfinite(values.median()) else 0.0
            self.numeric_medians[col] = median

        object_cols = [c for c in base_cols if df[c].dtype == object]
        object_cols = [c for c in object_cols if c not in self.drop_all_missing_cols and c not in self.very_sparse_cols]

        # 멀티라벨 컬럼: previous_class_* + 대표 다중선택 컬럼
        multilabel_hint = {
            "major_field",
            "desired_job",
            "certificate_acquisition",
            "desired_certificate",
            "desired_job_except_data",
            "interested_company",
            "expected_domain",
            "onedayclass_topic",
        }
        self.multilabel_cols = [c for c in base_cols if c.startswith("previous_class_")] + [
            c for c in object_cols if c in multilabel_hint
        ]
        self.multilabel_cols = sorted(set(self.multilabel_cols))

        # 텍스트 컬럼(서술형): 높은 cardinality + 길이 기반
        remaining_for_text = [c for c in object_cols if c not in self.multilabel_cols]
        self.text_cols = self._infer_text_cols(df, remaining_for_text)

        # 범주형 컬럼(나머지 object + 강제 범주형 numeric)
        categorical_cols: list[str] = []
        for col in base_cols:
            if col in self.drop_all_missing_cols or col in self.very_sparse_cols:
                continue
            if col in self.numeric_cols:
                continue
            if col in self.multilabel_cols:
                continue
            if col in self.text_cols:
                continue
            if df[col].dtype == object or col in force_categorical_numeric:
                categorical_cols.append(col)
        self.categorical_cols = categorical_cols

        # categorical counts
        self.cat_value_counts = {}
        self.cat_top_values = {}
        for col in self.categorical_cols:
            s = self._normalize_categorical(df[col])
            counts = s.value_counts(dropna=False).to_dict()
            self.cat_value_counts[col] = {str(k): int(v) for k, v in counts.items()}

            if not self.config.enable_categorical_onehot:
                continue
            nunique = int(s.nunique(dropna=False))
            if nunique <= 1:
                continue
            if nunique <= self.config.categorical_onehot_max_unique:
                top_k = min(self.config.categorical_onehot_top_k, nunique)
                self.cat_top_values[col] = list(s.value_counts().head(top_k).index.astype(str))

        # multilabel top tokens
        self.multilabel_top_tokens = {}
        if self.config.enable_multilabel_topk:
            for col in self.multilabel_cols:
                tokens, unique_tokens = self._split_multilabel(df[col])
                # document frequency 기준으로 top-k 선택
                df_counter: dict[str, int] = {}
                for items in unique_tokens:
                    for token in items:
                        df_counter[token] = df_counter.get(token, 0) + 1
                candidates = [(tok, c) for tok, c in df_counter.items() if c >= self.config.multilabel_min_df]
                candidates.sort(key=lambda x: (-x[1], x[0]))
                self.multilabel_top_tokens[col] = [tok for tok, _ in candidates[: self.config.multilabel_top_k]]

        # text models (T2)
        self._text_models = {}
        if self.config.text_mode.upper() == "T2" and self.text_cols:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            for col in self.text_cols:
                s = df[col].astype("string").fillna("").astype(str)
                vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)
                X = vectorizer.fit_transform(s)
                max_components = min(self.config.text_svd_components, X.shape[0] - 1, X.shape[1] - 1)
                if max_components <= 0:
                    continue
                svd = TruncatedSVD(n_components=max_components, random_state=self.config.cluster_random_state)
                svd.fit(X)
                self._text_models[col] = (vectorizer, svd)

        # clustering
        self._kmeans = None
        self._cluster_cols = []
        if self.config.enable_clustering:
            from sklearn.cluster import KMeans

            miss_cols = [c for c in base_cols if c not in {self.id_label, self.target_label}]
            miss_cols = [c for c in miss_cols if c not in self.drop_all_missing_cols]
            miss_cols = [c for c in miss_cols if 0.0 < missing_rate.get(c, 0.0) < 1.0]
            if len(miss_cols) >= 2 and self._n_train_rows >= self.config.cluster_k:
                X_miss = np.column_stack([self._is_missing(df[c]).astype(np.float32).to_numpy() for c in miss_cols])
                kmeans = KMeans(
                    n_clusters=int(self.config.cluster_k),
                    random_state=int(self.config.cluster_random_state),
                    n_init="auto",
                )
                kmeans.fit(X_miss)
                self._kmeans = kmeans
                self._cluster_cols = miss_cols

        # feature column order (fit 데이터 기준)
        sample = self.transform(df, keep_target=True)
        self.train_columns_ = sample.columns.tolist()
        self.inference_columns_ = [c for c in self.train_columns_ if c != self.target_label]

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, *, keep_target: bool) -> pd.DataFrame:
        if not self._raw_feature_cols:
            raise ValueError("Preprocessor가 초기화되지 않았습니다. fit을 먼저 호출하세요.")

        df = df.copy()
        for col in self._raw_feature_cols + [self.id_label] + ([self.target_label] if keep_target else []):
            if col not in df.columns:
                df[col] = np.nan

        out: dict[str, Any] = {}
        out[self.id_label] = df[self.id_label]
        if keep_target and self.target_label in df.columns:
            out[self.target_label] = df[self.target_label]

        # ---- 고차 피처: 응답 여부/참여도 집계 ----
        base_cols = [c for c in self._raw_feature_cols if c in df.columns]

        def answered(col: str) -> pd.Series:
            return (~self._is_missing(df[col])).astype(np.int8)

        # class1~4
        if self.class_cols:
            class_answered = np.column_stack([answered(c).to_numpy() for c in self.class_cols])
            out["num_class_choices"] = class_answered.sum(axis=1).astype(np.float32)
            for c in ["class2", "class3", "class4"]:
                if c in self.class_cols:
                    out[f"has_{_snake_case_name(c)}"] = answered(c).astype(np.float32)

        # previous_class_3~8
        if self.prev_cols:
            prev_answered = np.column_stack([answered(c).to_numpy() for c in self.prev_cols])
            out["has_prev_any"] = (prev_answered.sum(axis=1) > 0).astype(np.float32)
            out["num_prev_fields_answered"] = prev_answered.sum(axis=1).astype(np.float32)

            total_items = np.zeros(len(df), dtype=np.float32)
            all_unique: list[set[str]] = [set() for _ in range(len(df))]
            for c in self.prev_cols:
                tokens, unique_tokens = self._split_multilabel(df[c])
                total_items += unique_tokens.apply(len).astype(np.float32).to_numpy()
                for i, items in enumerate(unique_tokens):
                    all_unique[i].update(items)
            out["num_prev_items_total"] = total_items
            out["num_prev_unique_total"] = np.array([len(s) for s in all_unique], dtype=np.float32)

        # 전체/그룹별 응답 개수
        answered_matrix = np.column_stack([answered(c).to_numpy() for c in base_cols]) if base_cols else np.zeros((len(df), 0))
        out["num_answered_total"] = answered_matrix.sum(axis=1).astype(np.float32)
        for group_name, cols in (self.answer_groups or {}).items():
            cols = [c for c in cols if c in df.columns]
            if not cols:
                continue
            mat = np.column_stack([answered(c).to_numpy() for c in cols])
            out[f"num_answered_{_snake_case_name(group_name)}"] = mat.sum(axis=1).astype(np.float32)

        # ---- 매우 sparse 컬럼: 원본 drop, 답변여부/개수만 ----
        for col in self.very_sparse_cols:
            prefix = self._prefix.get(col, _snake_case_name(col))
            is_ans = answered(col).astype(np.float32)
            out[f"{prefix}_is_answered"] = is_ans
            if df[col].dtype == object:
                _, unique_tokens = self._split_multilabel(df[col])
                out[f"{prefix}_num_items"] = unique_tokens.apply(len).astype(np.float32)
                out[f"{prefix}_num_unique"] = unique_tokens.apply(len).astype(np.float32)

        # ---- numeric: median impute + is_missing ----
        for col in self.numeric_cols:
            prefix = self._prefix.get(col, _snake_case_name(col))
            missing = self._is_missing(df[col])
            values = pd.to_numeric(df[col], errors="coerce")
            values = values.where(~missing, np.nan).fillna(self.numeric_medians.get(col, 0.0)).astype(np.float32)
            out[prefix] = values
            out[f"is_missing_{prefix}"] = missing.astype(np.float32)

        # ---- categorical: freq/log_freq/is_rare (+ optional top-k one-hot) ----
        for col in self.categorical_cols:
            prefix = self._prefix.get(col, _snake_case_name(col))
            s = self._normalize_categorical(df[col])

            counts = self.cat_value_counts.get(col, {})
            count_arr = s.map(lambda v: float(counts.get(str(v), 0))).astype(np.float32)
            freq = count_arr / float(max(self._n_train_rows, 1))
            out[f"{prefix}_freq"] = freq
            out[f"{prefix}_log_freq"] = np.log1p(count_arr).astype(np.float32)
            out[f"{prefix}_is_rare"] = (count_arr < float(self.config.rare_threshold)).astype(np.float32)

            top_values = self.cat_top_values.get(col) or []
            for i, v in enumerate(top_values):
                out[f"{prefix}_oh_{i}"] = (s.astype(str) == str(v)).astype(np.float32)

        # ---- multilabel: 집계형 + optional top-k token binarization ----
        for col in self.multilabel_cols:
            prefix = self._prefix.get(col, _snake_case_name(col))
            tokens, unique_tokens = self._split_multilabel(df[col])
            out[f"{prefix}_has_any"] = unique_tokens.apply(lambda xs: float(len(xs) > 0)).astype(np.float32)
            out[f"{prefix}_num_items"] = tokens.apply(len).astype(np.float32)
            out[f"{prefix}_num_unique"] = unique_tokens.apply(len).astype(np.float32)

            top_tokens = self.multilabel_top_tokens.get(col) or []
            if top_tokens:
                def as_set(items: list[str]) -> set[str]:
                    return set(items)

                sets = unique_tokens.apply(as_set)
                for i, tok in enumerate(top_tokens):
                    out[f"{prefix}_tok_{i}"] = sets.apply(lambda s: float(tok in s)).astype(np.float32)

        # ---- text: T0/T1/T2 ----
        text_mode = (self.config.text_mode or "T1").upper()
        for col in self.text_cols:
            prefix = self._prefix.get(col, _snake_case_name(col))
            s = df[col].astype("string").fillna("").astype(str)
            if text_mode == "T0":
                continue

            out[f"{prefix}_len"] = s.str.len().astype(np.float32)
            out[f"{prefix}_num_tokens"] = s.str.split().apply(len).astype(np.float32)
            out[f"{prefix}_has_number"] = s.str.contains(r"\\d", regex=True).astype(np.float32)
            out[f"{prefix}_punct_count"] = s.str.count(r"[\\.,!\\?;:\\-\\(\\)\\[\\]\\{\\}\\/\\\\'\\\"]").astype(np.float32)

            if text_mode == "T2":
                model = self._text_models.get(col)
                if model is None:
                    continue
                vectorizer, svd = model
                X = vectorizer.transform(s)
                emb = svd.transform(X).astype(np.float32)
                for j in range(emb.shape[1]):
                    out[f"{prefix}_svd_{j}"] = emb[:, j]

        # ---- clustering on missingness pattern ----
        if self._kmeans is not None and self._cluster_cols:
            X_miss = np.column_stack([self._is_missing(df[c]).astype(np.float32).to_numpy() for c in self._cluster_cols])
            cluster_id = self._kmeans.predict(X_miss).astype(np.int32)
            out["missing_cluster_id"] = cluster_id.astype(np.float32)
            dist = self._kmeans.transform(X_miss).astype(np.float32)
            for k in range(dist.shape[1]):
                out[f"missing_cluster_dist_{k}"] = dist[:, k]
            # low-dimensional one-hot
            for k in range(dist.shape[1]):
                out[f"missing_cluster_oh_{k}"] = (cluster_id == k).astype(np.float32)

        processed = pd.DataFrame(out)

        if self.config.drop_constant_features:
            keep_cols = [self.id_label]
            if keep_target:
                keep_cols.append(self.target_label)
            for col in processed.columns:
                if col in keep_cols:
                    continue
                if processed[col].nunique(dropna=False) <= 1:
                    continue
                keep_cols.append(col)
            processed = processed[keep_cols]

        expected_cols = self.train_columns_ if keep_target else self.inference_columns_
        if expected_cols:
            # 컬럼 순서 고정 + 누락 컬럼 0으로 채우기
            for col in expected_cols:
                if col not in processed.columns:
                    processed[col] = 0.0
            processed = processed[expected_cols]

        # 안전장치: 비수치형이 남으면 해시로 강제 변환
        non_numeric = processed.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
        non_numeric = [c for c in non_numeric if c not in {self.id_label, self.target_label}]
        for col in non_numeric:
            hashed = pd.util.hash_pandas_object(processed[col].astype(str), index=False)
            processed[col] = (hashed % 1_000_000).astype(np.float32)

        numeric_cols = [c for c in processed.columns if c not in {self.id_label, self.target_label}]
        processed[numeric_cols] = processed[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return processed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df, keep_target=True)


_DEFAULT_PREPROCESSOR: Optional[TabularPreprocessor] = None


def preprocess_data(
    file_path: str,
    is_train: bool = True,
    train_cols: Optional[list[str]] = None,
    id_label: str = "ID",
    target_label: str = "completed",
    *,
    preprocessor: Optional[TabularPreprocessor] = None,
    config: Optional[PreprocessConfig] = None,
):
    df = pd.read_csv(file_path)

    if is_train:
        missing_cols = [c for c in [id_label, target_label] if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Required columns missing from training data: {missing_cols}")

        y = df[target_label]
        fitted = preprocessor or TabularPreprocessor(id_label=id_label, target_label=target_label, config=config)
        processed = fitted.fit_transform(df)

        global _DEFAULT_PREPROCESSOR
        _DEFAULT_PREPROCESSOR = fitted

        return processed, y, [], []

    ids = df[id_label] if id_label in df.columns else None

    fitted = preprocessor or _DEFAULT_PREPROCESSOR
    if fitted is None:
        raise ValueError(
            "전처리기가 없습니다. 같은 프로세스에서 먼저 `preprocess_data(..., is_train=True)`를 호출하거나 "
            "`preprocessor=`로 학습 전처리기를 전달하세요."
        )

    processed = fitted.transform(df, keep_target=False)
    return processed, ids, [], []
