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

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Any, Optional

import numpy as np
import pandas as pd


_MISSING_STRINGS = {
    "__missing__",
    "nan",
    "none",
    "null",
}

_DEFAULT_NONE_STRINGS = {
    "해당없음",
    "해당 없음",
    "없음",
    "없습니다",
    "무",
}

_DEFAULT_NONE_STRINGS_LOWER = {t.lower() for t in _DEFAULT_NONE_STRINGS}

_MAJOR_LABEL_IT = "IT(컴퓨터 공학 포함)"
_MAJOR_LABELS = (
    _MAJOR_LABEL_IT,
    "경영학",
    "경제통상학",
    "자연과학",
    "사회과학",
    "인문학",
    "의약학",
    "교육학",
    "법학",
    "예체능",
    "기타",
)

_DEFAULT_MAJOR_KEYWORDS: dict[str, list[str]] = {
    _MAJOR_LABEL_IT: [
        "컴퓨터",
        "소프트웨어",
        "정보",
        "데이터",
        "ai",
        "인공지능",
        "전산",
        "전자",
        "통신",
        "빅데이터",
        "공학",
        "산업공학",
        "인공지능학",
        "소프트웨어학",
    ],
    "경영학": ["경영", "회계", "경영정보", "재무", "마케팅"],
    "경제통상학": ["경제", "통상", "무역", "국제통상"],
    "자연과학": ["수학", "통계", "물리", "화학", "생명", "지구", "환경"],
    "사회과학": ["심리", "사회", "정치", "행정", "국제", "커뮤니케이션", "언론", "미디어"],
    "인문학": ["국문", "영문", "문헌", "사학", "철학", "언어", "역사"],
    "의약학": ["의", "약", "간호", "보건", "치의", "한의", "수의"],
    "교육학": ["교육", "교직"],
    "법학": ["법"],
    "예체능": ["예술", "체육", "음악", "미술", "디자인", "연극", "영화"],
    "기타": [],
}

_DEFAULT_MAJOR_PRIORITY = [
    "의약학",
    "법학",
    "교육학",
    "예체능",
    _MAJOR_LABEL_IT,
    "경영학",
    "경제통상학",
    "자연과학",
    "사회과학",
    "인문학",
    "기타",
]

_DEFAULT_CERTIFICATE_SLOTS: dict[str, list[str]] = {
    "adsp": ["adsp", "데이터분석준전문가"],
    "sqld": ["sqld", "sql개발자", "sql 개발자"],
    "정보처리기사": ["정보처리기사"],
    "빅데이터분석기사": ["빅데이터분석기사"],
    "컴퓨터활용능력": ["컴퓨터활용능력", "컴활"],
    "토익": ["toeic", "토익"],
    "오픽": ["opic", "오픽"],
    "태블로": ["tableau", "태블로"],
    "구글애널리틱스": ["googleanalytics", "구글애널리틱", "ga"],
    "aws": ["aws"],
}

_DEFAULT_PREV_TEXT_KEYWORDS: dict[str, list[str]] = {
    "python": ["파이썬", "python"],
    "sql": ["sql"],
    "pandas": ["판다스", "pandas"],
    "ml": ["머신러닝", "딥러닝", "machine", "deep"],
    "stat": ["통계", "statistics"],
    "preprocess": ["전처리"],
}


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
    multilabel_parser_rules: list[dict[str, str]] = field(default_factory=list)

    enable_none_features: bool = True
    none_strings: list[str] = field(default_factory=lambda: sorted(_DEFAULT_NONE_STRINGS))

    text_mode: str = "T1"  # "T0"|"T1"|"T2"
    text_svd_components: int = 64
    enable_multilabel_svd: bool = False
    multilabel_svd_cols: list[str] = field(
        default_factory=lambda: ["interested_company", "certificate_acquisition", "desired_certificate"]
    )
    multilabel_svd_components: int = 32
    multilabel_svd_min_df: int = 2

    enable_major_mapping: bool = True
    major_mapping_ruleset: str = "v1"
    major_keyword_rules: dict[str, list[str]] = field(default_factory=lambda: dict(_DEFAULT_MAJOR_KEYWORDS))
    major_priority: list[str] = field(default_factory=lambda: list(_DEFAULT_MAJOR_PRIORITY))
    major_other_label: str = "OTHER"

    completed_semester_valid_min: float = 0.0
    completed_semester_valid_max: float = 20.0
    completed_semester_invalid_to_nan: bool = True

    enable_certificate_slots: bool = True
    certificate_slots: dict[str, list[str]] = field(default_factory=lambda: dict(_DEFAULT_CERTIFICATE_SLOTS))

    enable_prev_code_agg_topk: bool = True
    prev_code_top_k: int = 20
    prev_code_min_df: int = 2
    enable_prev_text_keywords: bool = True
    prev_text_keyword_rules: dict[str, list[str]] = field(default_factory=lambda: dict(_DEFAULT_PREV_TEXT_KEYWORDS))

    enable_clustering: bool = True
    cluster_k: int = 5
    cluster_ks: list[int] = field(default_factory=list)
    cluster_random_state: int = 42
    enable_clustering_gmm: bool = False
    gmm_k: int = 5

    enable_transductive_fit: bool = False
    transductive_fit_csv: str = ""
    enable_domain_adaptation_feature: bool = False
    domain_adaptation_feature_name: str = "p_test"
    domain_adaptation_C: float = 1.0
    domain_adaptation_max_iter: int = 300

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

        self._original_feature_cols: list[str] = []
        self._derived_feature_cols: list[str] = []
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
        self.prev_top_codes: list[str] = []

        self.text_cols: list[str] = []
        self._text_models: dict[str, tuple[Any, Any]] = {}
        self._multilabel_svd_models: dict[str, tuple[Any, Any]] = {}

        self._cluster_cols: list[str] = []
        self._primary_cluster_k: Optional[int] = None
        self._kmeans_by_k: dict[int, Any] = {}
        self._gmm: Any = None
        self._domain_scaler: Any = None
        self._domain_clf: Any = None
        self._domain_feature_cols: list[str] = []

        self.train_columns_: list[str] = []
        self.inference_columns_: list[str] = []
        self._n_train_rows: int = 0

        self._none_strings_lower: set[str] = {str(v).strip().lower() for v in (self.config.none_strings or [])}
        if not self._none_strings_lower:
            self._none_strings_lower = set(_DEFAULT_NONE_STRINGS_LOWER)

        self._compiled_parser_rules: list[tuple[re.Pattern[str], str]] = []
        for rule in self.config.multilabel_parser_rules or []:
            try:
                pattern = str(rule.get("match", "")).strip()
                parser = str(rule.get("parser", "")).strip()
                if not pattern or not parser:
                    continue
                self._compiled_parser_rules.append((re.compile(pattern), parser))
            except Exception:
                continue

    def _is_missing(self, series: pd.Series) -> pd.Series:
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            s = series.astype("string")
            lowered = s.str.strip().str.lower()
            return series.isna() | lowered.eq("") | lowered.isin(_MISSING_STRINGS)
        return series.isna()

    def _normalize_text(self, value: Any) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass

        text = unicodedata.normalize("NFKC", str(value))
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _is_none(self, series: pd.Series) -> pd.Series:
        if not self.config.enable_none_features:
            return pd.Series(False, index=series.index)
        if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)):
            return pd.Series(False, index=series.index)
        missing = self._is_missing(series)
        lowered = series.astype("string").str.strip().str.lower()
        return (~missing) & lowered.isin(self._none_strings_lower)

    def _normalize_categorical(self, series: pd.Series) -> pd.Series:
        missing = self._is_missing(series)
        none = self._is_none(series)
        s = series.astype("string").fillna("").apply(self._normalize_text)
        s = s.where(~missing, "__MISSING__")
        if self.config.enable_none_features:
            s = s.where(~none, "__NONE__")
        return s.fillna("__MISSING__")

    def _normalize_token(self, token: str) -> str:
        token = self._normalize_text(token)
        if not token:
            return ""
        lowered = token.lower()
        if lowered in _MISSING_STRINGS:
            return ""
        if lowered in self._none_strings_lower:
            return ""

        m = re.match(r"^(\d{4})\s*[:：]", token)
        if m:
            return m.group(1)

        m = re.match(r"^([A-Za-z])\s*\.", token)
        if m:
            return m.group(1).upper()

        return token

    def _select_multilabel_parser(self, col: str) -> str:
        for pattern, parser in self._compiled_parser_rules:
            if pattern.search(col):
                return parser
        if col.startswith("previous_class_"):
            return "code4_prefix"
        if col in {"desired_job", "desired_job_except_data", "expected_domain"}:
            return "alpha_dot"
        if col == "onedayclass_topic":
            return "comma_outside_parens"
        if col in {"certificate_acquisition", "desired_certificate"}:
            return "certificate_normalize"
        return "comma"

    def _split_by_comma_outside_parens(self, text: str) -> list[str]:
        parts: list[str] = []
        buf: list[str] = []
        depth = 0
        for ch in text:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(depth - 1, 0)
            if ch == "," and depth == 0:
                parts.append("".join(buf).strip())
                buf = []
                continue
            buf.append(ch)
        parts.append("".join(buf).strip())
        return [p for p in parts if p]

    def _remove_parens_content(self, text: str) -> str:
        out: list[str] = []
        depth = 0
        for ch in text:
            if ch == "(":
                depth += 1
                continue
            if ch == ")":
                depth = max(depth - 1, 0)
                continue
            if depth == 0:
                out.append(ch)
        return "".join(out)

    def _normalize_certificate_token(self, token: str) -> str:
        token = self._normalize_text(token)
        if not token:
            return ""
        lowered = token.lower()
        if lowered in _MISSING_STRINGS:
            return ""
        if lowered in self._none_strings_lower:
            return ""

        token = re.sub(r"^\s*(준비중|준비 중)\s*[:：]\s*", "", token)
        token = self._remove_parens_content(token)
        token = token.replace(" ", "")
        token = re.sub(r"[·\t\r\n]", "", token)
        return token

    def _split_multilabel(self, col: str, series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        missing = self._is_missing(series)
        none_present = self._is_none(series)
        s = series.astype("string").fillna("").astype(str)
        parser = self._select_multilabel_parser(col)

        def to_tokens(value: str) -> list[str]:
            value = self._normalize_text(value)
            if not value:
                return []
            if parser == "code4_prefix":
                return [m.group(1) for m in re.finditer(r"(\d{4})\s*[:：]", value)]
            if parser == "alpha_dot":
                return [m.group(1).upper() for m in re.finditer(r"([A-Za-z])\s*\.", value)]
            if parser == "comma_outside_parens":
                parts = self._split_by_comma_outside_parens(value)
                parts = [self._remove_parens_content(p).strip() for p in parts]
                normalized = [self._normalize_token(p) for p in parts]
                return [t for t in normalized if t]
            if parser == "certificate_normalize":
                parts = self._split_by_comma_outside_parens(value)
                normalized = [self._normalize_certificate_token(p) for p in parts]
                return [t for t in normalized if t]

            raw_tokens = [t.strip() for t in value.split(",")]
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
        is_none = none_present & unique.apply(lambda xs: len(xs) == 0) & (~missing)
        is_none_conflict = none_present & unique.apply(lambda xs: len(xs) > 0) & (~missing)
        return tokens, unique, is_none.astype(np.int8), is_none_conflict.astype(np.int8)

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

    def _map_major_value_to_coarse(self, value: Any) -> tuple[str, int, int]:
        text = self._normalize_text(value)
        if not text:
            return "__MISSING__", 0, 0
        lowered = text.strip().lower()
        if lowered in _MISSING_STRINGS:
            return "__MISSING__", 0, 0
        if lowered in self._none_strings_lower:
            return "__NONE__", 0, 0

        for label in _MAJOR_LABELS:
            if label == text:
                return label, 0, 0

        cleaned = re.sub(r"(학과|학부|전공)$", "", text).strip()
        cleaned_lower = cleaned.lower()

        rules = self.config.major_keyword_rules or {}
        matched: list[str] = []
        for label, keywords in rules.items():
            if not keywords:
                continue
            for kw in keywords:
                kw_norm = str(kw).strip().lower()
                if not kw_norm:
                    continue
                if kw_norm in cleaned_lower:
                    matched.append(str(label))
                    break

        matched_unique = list(dict.fromkeys(matched))
        if not matched_unique:
            other = str(self.config.major_other_label or "OTHER").strip() or "OTHER"
            return other, 1, 0

        priority = self.config.major_priority or []
        chosen = None
        for p in priority:
            if str(p) in matched_unique:
                chosen = str(p)
                break
        if chosen is None:
            chosen = matched_unique[0]

        ambiguous = 1 if len(matched_unique) > 1 else 0
        return chosen, 0, ambiguous

    def _add_major_coarse_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        derived: list[str] = []
        for src, dst in (("major1_1", "major1_1_coarse"), ("major1_2", "major1_2_coarse")):
            if src not in df.columns:
                continue
            mapped = df[src].astype("string").apply(self._map_major_value_to_coarse)
            df[dst] = mapped.apply(lambda x: x[0])
            derived.append(dst)
        self._derived_feature_cols = sorted(set(self._derived_feature_cols + derived))
        return df

    def _clean_completed_semester(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        missing = self._is_missing(series)
        values = pd.to_numeric(series.where(~missing, np.nan), errors="coerce")
        values = values.astype(float)
        valid_min = float(self.config.completed_semester_valid_min)
        valid_max = float(self.config.completed_semester_valid_max)

        int_like = np.isfinite(values.to_numpy()) & (np.abs(values.to_numpy() - np.round(values.to_numpy())) < 1e-6)
        int_like = pd.Series(int_like, index=series.index)
        outlier = (~missing) & values.notna() & ((values < valid_min) | (values > valid_max) | (~int_like))

        cleaned = values.copy()
        if self.config.completed_semester_invalid_to_nan:
            cleaned = cleaned.mask(outlier, np.nan)
        else:
            cleaned = cleaned.clip(lower=valid_min, upper=valid_max).round()
        return cleaned, outlier.astype(np.int8)

    def fit(self, df: pd.DataFrame, df_unlabeled: Optional[pd.DataFrame] = None) -> "TabularPreprocessor":
        if self.id_label not in df.columns:
            raise ValueError(f"id_label({self.id_label}) 컬럼이 없습니다.")
        if self.target_label not in df.columns:
            raise ValueError(f"target_label({self.target_label}) 컬럼이 없습니다.")

        df_train = df.copy()
        self._n_train_rows = int(len(df_train))

        self._original_feature_cols = [c for c in df_train.columns if c not in {self.id_label, self.target_label}]
        self._derived_feature_cols = []
        if self.config.enable_major_mapping:
            df_train = self._add_major_coarse_columns(df_train)

        base_cols = [c for c in df_train.columns if c not in {self.id_label, self.target_label}]
        self._raw_feature_cols = list(base_cols)
        self._prefix = _ensure_unique_prefixes(base_cols)

        df_fit = df_train
        if self.config.enable_transductive_fit and df_unlabeled is not None and len(df_unlabeled) > 0:
            df_u = df_unlabeled.copy()
            if self.id_label not in df_u.columns:
                df_u[self.id_label] = np.arange(len(df_u))
            if self.target_label not in df_u.columns:
                df_u[self.target_label] = np.nan
            if self.config.enable_major_mapping:
                df_u = self._add_major_coarse_columns(df_u)
            for col in base_cols:
                if col not in df_u.columns:
                    df_u[col] = np.nan
            df_fit = pd.concat(
                [df_train[[self.id_label, self.target_label] + base_cols], df_u[[self.id_label, self.target_label] + base_cols]],
                axis=0,
                ignore_index=True,
            )

        missing_rate = {c: float(self._is_missing(df_train[c]).mean()) for c in base_cols}
        self.drop_all_missing_cols = [c for c, r in missing_rate.items() if r >= 1.0]
        self.very_sparse_cols = [
            c for c, r in missing_rate.items() if (r >= self.config.very_sparse_threshold and r < 1.0)
        ]

        self.prev_cols = [c for c in self._original_feature_cols if c.startswith("previous_class_")]
        self.class_cols = [c for c in ["class1", "class2", "class3", "class4"] if c in self._original_feature_cols]

        self.answer_groups = {
            "prev": self.prev_cols,
            "class": self.class_cols,
            "job": [
                c
                for c in ["job", "desired_job", "desired_job_except_data", "desired_career_path"]
                if c in self._original_feature_cols
            ],
            "company": [c for c in ["interested_company", "incumbents_company_level"] if c in self._original_feature_cols],
            "cert": [c for c in ["certificate_acquisition", "desired_certificate"] if c in self._original_feature_cols],
            "major": [
                c
                for c in ["major type", "major1_1", "major1_2", "major_field", "major_data"]
                if c in self._original_feature_cols
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
            and df_train[c].dtype.kind in {"i", "u", "f", "b"}
        ]
        self.numeric_cols = list(numeric_candidates)

        # numeric median
        self.numeric_medians = {}
        for col in self.numeric_cols:
            if col == "completed_semester":
                values, _ = self._clean_completed_semester(df_train[col])
            else:
                missing = self._is_missing(df_train[col])
                values = pd.to_numeric(df_train[col].where(~missing, np.nan), errors="coerce")
            median = float(values.median()) if np.isfinite(values.median()) else 0.0
            self.numeric_medians[col] = median

        object_cols = [
            c
            for c in base_cols
            if pd.api.types.is_string_dtype(df_train[c]) or pd.api.types.is_object_dtype(df_train[c])
        ]
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
        self.multilabel_cols = list(self.prev_cols) + [c for c in object_cols if c in multilabel_hint]
        self.multilabel_cols = sorted(set(self.multilabel_cols))

        # 텍스트 컬럼(서술형): 높은 cardinality + 길이 기반
        remaining_for_text = [c for c in object_cols if c not in self.multilabel_cols]
        self.text_cols = self._infer_text_cols(df_train, remaining_for_text)

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
            if pd.api.types.is_string_dtype(df_train[col]) or pd.api.types.is_object_dtype(df_train[col]) or col in force_categorical_numeric:
                categorical_cols.append(col)
        self.categorical_cols = categorical_cols

        # categorical counts
        self.cat_value_counts = {}
        self.cat_top_values = {}
        for col in self.categorical_cols:
            s = self._normalize_categorical(df_train[col])
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
                if col not in df_fit.columns:
                    continue
                _, unique_tokens, _, _ = self._split_multilabel(col, df_fit[col])
                # document frequency 기준으로 top-k 선택
                df_counter: dict[str, int] = {}
                for items in unique_tokens:
                    for token in items:
                        df_counter[token] = df_counter.get(token, 0) + 1
                candidates = [(tok, c) for tok, c in df_counter.items() if c >= self.config.multilabel_min_df]
                candidates.sort(key=lambda x: (-x[1], x[0]))
                self.multilabel_top_tokens[col] = [tok for tok, _ in candidates[: self.config.multilabel_top_k]]

        # previous_class aggregated top codes
        self.prev_top_codes = []
        if self.config.enable_prev_code_agg_topk and self.prev_cols:
            row_sets: list[set[str]] = [set() for _ in range(len(df_fit))]
            for col in self.prev_cols:
                if col not in df_fit.columns:
                    continue
                _, unique_tokens, _, _ = self._split_multilabel(col, df_fit[col])
                for i, items in enumerate(unique_tokens):
                    row_sets[i].update(items)
            df_counter: dict[str, int] = {}
            for items in row_sets:
                for token in items:
                    df_counter[token] = df_counter.get(token, 0) + 1
            candidates = [(tok, c) for tok, c in df_counter.items() if c >= self.config.prev_code_min_df]
            candidates.sort(key=lambda x: (-x[1], x[0]))
            self.prev_top_codes = [tok for tok, _ in candidates[: self.config.prev_code_top_k]]

        # text models (T2)
        self._text_models = {}
        if self.config.text_mode.upper() == "T2" and self.text_cols:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            for col in self.text_cols:
                s = df_fit[col].astype("string").fillna("").astype(str)
                vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2)
                X = vectorizer.fit_transform(s)
                max_components = min(self.config.text_svd_components, X.shape[0] - 1, X.shape[1] - 1)
                if max_components <= 0:
                    continue
                svd = TruncatedSVD(n_components=max_components, random_state=self.config.cluster_random_state)
                svd.fit(X)
                self._text_models[col] = (vectorizer, svd)

        # multilabel token TF-IDF -> SVD (옵션)
        self._multilabel_svd_models = {}
        if self.config.enable_multilabel_svd:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            for col in self.config.multilabel_svd_cols or []:
                if col not in self.multilabel_cols or col not in df_fit.columns:
                    continue
                _, unique_tokens, _, _ = self._split_multilabel(col, df_fit[col])
                docs = unique_tokens.apply(lambda xs: " ".join(xs))
                vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\\b\\w+\\b", min_df=int(self.config.multilabel_svd_min_df))
                X = vectorizer.fit_transform(docs)
                max_components = min(self.config.multilabel_svd_components, X.shape[0] - 1, X.shape[1] - 1)
                if max_components <= 0:
                    continue
                svd = TruncatedSVD(n_components=max_components, random_state=self.config.cluster_random_state)
                svd.fit(X)
                self._multilabel_svd_models[col] = (vectorizer, svd)

        # clustering
        self._primary_cluster_k = None
        self._kmeans_by_k = {}
        self._cluster_cols = []
        self._gmm = None
        if self.config.enable_clustering:
            from sklearn.cluster import KMeans

            miss_cols = [c for c in self._original_feature_cols if c not in self.drop_all_missing_cols]
            miss_cols = [c for c in miss_cols if 0.0 < missing_rate.get(c, 0.0) < 1.0]
            if len(miss_cols) >= 2:
                X_miss = np.column_stack([self._is_missing(df_fit[c]).astype(np.float32).to_numpy() for c in miss_cols])
                k_list = [int(self.config.cluster_k)] + [int(k) for k in (self.config.cluster_ks or [])]
                k_list = sorted({k for k in k_list if k >= 2})
                for k in k_list:
                    if X_miss.shape[0] < k:
                        continue
                    kmeans = KMeans(
                        n_clusters=int(k),
                        random_state=int(self.config.cluster_random_state),
                        n_init="auto",
                    )
                    kmeans.fit(X_miss)
                    self._kmeans_by_k[int(k)] = kmeans
                if self._kmeans_by_k:
                    primary = int(self.config.cluster_k)
                    if primary not in self._kmeans_by_k:
                        primary = sorted(self._kmeans_by_k.keys())[0]
                    self._primary_cluster_k = primary
                    self._cluster_cols = miss_cols

                if self.config.enable_clustering_gmm:
                    try:
                        from sklearn.mixture import GaussianMixture

                        gmm_k = int(self.config.gmm_k)
                        if X_miss.shape[0] >= gmm_k and gmm_k >= 2:
                            gmm = GaussianMixture(n_components=gmm_k, random_state=int(self.config.cluster_random_state))
                            gmm.fit(X_miss)
                            self._gmm = gmm
                    except Exception:
                        self._gmm = None

        # domain adaptation (train vs test) feature (옵션)
        self._domain_scaler = None
        self._domain_clf = None
        self._domain_feature_cols = []
        if self.config.enable_domain_adaptation_feature and df_unlabeled is not None and len(df_unlabeled) > 0:
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler

                df_u = df_unlabeled.copy()
                if self.id_label not in df_u.columns:
                    df_u[self.id_label] = np.arange(len(df_u))
                if self.target_label not in df_u.columns:
                    df_u[self.target_label] = np.nan
                if self.config.enable_major_mapping:
                    df_u = self._add_major_coarse_columns(df_u)
                for col in base_cols:
                    if col not in df_u.columns:
                        df_u[col] = np.nan

                train_processed = self.transform(df_train, keep_target=True)
                test_processed = self.transform(df_u, keep_target=False)
                feature_cols = [c for c in train_processed.columns if c not in {self.id_label, self.target_label}]

                X_train = train_processed[feature_cols].to_numpy(dtype=np.float32, copy=True)
                X_test = test_processed.reindex(columns=feature_cols, fill_value=0.0).to_numpy(dtype=np.float32, copy=True)

                X_all = np.vstack([X_train, X_test])
                y_all = np.concatenate([np.zeros(len(X_train), dtype=np.int32), np.ones(len(X_test), dtype=np.int32)])

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_all)
                clf = LogisticRegression(
                    max_iter=int(self.config.domain_adaptation_max_iter),
                    C=float(self.config.domain_adaptation_C),
                    solver="lbfgs",
                )
                clf.fit(X_scaled, y_all)
                self._domain_scaler = scaler
                self._domain_clf = clf
                self._domain_feature_cols = feature_cols
            except Exception:
                self._domain_scaler = None
                self._domain_clf = None
                self._domain_feature_cols = []

        # feature column order (fit 데이터 기준)
        sample = self.transform(df_train, keep_target=True)
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

        major_map_failed: dict[str, pd.Series] = {}
        major_map_ambiguous: dict[str, pd.Series] = {}
        if self.config.enable_major_mapping:
            for src, dst in (("major1_1", "major1_1_coarse"), ("major1_2", "major1_2_coarse")):
                if src not in df.columns:
                    continue
                mapped = df[src].astype("string").apply(self._map_major_value_to_coarse)
                df[dst] = mapped.apply(lambda x: x[0])
                major_map_failed[dst] = mapped.apply(lambda x: x[1]).astype(np.float32)
                major_map_ambiguous[dst] = mapped.apply(lambda x: x[2]).astype(np.float32)

        out: dict[str, Any] = {}
        out[self.id_label] = df[self.id_label]
        if keep_target and self.target_label in df.columns:
            out[self.target_label] = df[self.target_label]

        # ---- 고차 피처: 응답 여부/참여도 집계 ----
        base_cols = [c for c in (self._original_feature_cols or []) if c in df.columns]

        def answered(col: str) -> pd.Series:
            return (~self._is_missing(df[col])).astype(np.int8)

        def missing(col: str) -> pd.Series:
            return self._is_missing(df[col]).astype(np.int8)

        def is_none(col: str) -> pd.Series:
            return self._is_none(df[col]).astype(np.int8)

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
            prev_none_any = np.zeros(len(df), dtype=np.float32)
            for c in self.prev_cols:
                tokens, unique_tokens, none_flag, _ = self._split_multilabel(c, df[c])
                total_items += tokens.apply(len).astype(np.float32).to_numpy()
                prev_none_any = np.maximum(prev_none_any, none_flag.astype(np.float32).to_numpy())
                for i, items in enumerate(unique_tokens):
                    all_unique[i].update(items)
            prev_unique_cnt = np.array([len(s) for s in all_unique], dtype=np.float32)

            out["prev_none_any"] = prev_none_any
            out["prev_taken_cnt"] = total_items
            out["prev_unique_code_cnt"] = prev_unique_cnt
            out["num_prev_items_total"] = total_items
            out["num_prev_unique_total"] = prev_unique_cnt

            if self.prev_top_codes:
                for code in self.prev_top_codes:
                    out[f"prev_has_{code}"] = np.array([float(code in s) for s in all_unique], dtype=np.float32)

            if self.config.enable_prev_text_keywords and (self.config.prev_text_keyword_rules or {}):
                prev_text_cols: list[pd.Series] = []
                for c in self.prev_cols:
                    raw = df[c].astype("string").fillna("").apply(self._normalize_text).str.lower()
                    valid = (~self._is_missing(df[c])) & (~self._is_none(df[c]))
                    prev_text_cols.append(raw.where(valid, ""))

                for kw_name, patterns in (self.config.prev_text_keyword_rules or {}).items():
                    pats = [self._normalize_text(p).lower() for p in (patterns or [])]
                    pats = [p for p in pats if p]
                    if not pats:
                        continue

                    hits: list[pd.Series] = []
                    for col_txt in prev_text_cols:
                        hit = pd.Series(False, index=df.index)
                        for p in pats:
                            hit = hit | col_txt.str.contains(p, regex=False)
                        hits.append(hit.astype(np.int8))

                    cnt = np.column_stack([h.to_numpy() for h in hits]).sum(axis=1).astype(np.float32)
                    safe = _snake_case_name(kw_name)
                    out[f"prev_kw_cnt_{safe}"] = cnt
                    out[f"prev_kw_has_{safe}"] = (cnt > 0).astype(np.float32)

        # 전체/그룹별 응답 개수
        answered_matrix = (
            np.column_stack([answered(c).to_numpy() for c in base_cols]) if base_cols else np.zeros((len(df), 0))
        )
        answered_total = answered_matrix.sum(axis=1).astype(np.float32)
        out["num_answered_total"] = answered_total
        out["num_missing_total"] = (float(len(base_cols)) - answered_total).astype(np.float32)
        if self.config.enable_none_features:
            none_matrix = (
                np.column_stack([is_none(c).to_numpy() for c in base_cols]) if base_cols else np.zeros((len(df), 0))
            )
            out["num_none_total"] = none_matrix.sum(axis=1).astype(np.float32)
        for group_name, cols in (self.answer_groups or {}).items():
            cols = [c for c in cols if c in df.columns]
            if not cols:
                continue
            mat = np.column_stack([answered(c).to_numpy() for c in cols])
            group_answered = mat.sum(axis=1).astype(np.float32)
            out[f"num_answered_{_snake_case_name(group_name)}"] = group_answered
            out[f"num_missing_{_snake_case_name(group_name)}"] = (float(len(cols)) - group_answered).astype(np.float32)
            if self.config.enable_none_features:
                none_mat = np.column_stack([is_none(c).to_numpy() for c in cols])
                out[f"num_none_{_snake_case_name(group_name)}"] = none_mat.sum(axis=1).astype(np.float32)

        # ---- 매우 sparse 컬럼: 원본 drop, 답변여부/개수만 ----
        for col in self.very_sparse_cols:
            prefix = self._prefix.get(col, _snake_case_name(col))
            is_ans = answered(col).astype(np.float32)
            out[f"{prefix}_is_answered"] = is_ans
            out[f"is_missing_{prefix}"] = missing(col).astype(np.float32)
            if self.config.enable_none_features:
                out[f"is_none_{prefix}"] = is_none(col).astype(np.float32)
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                tokens, unique_tokens, none_flag, none_conflict = self._split_multilabel(col, df[col])
                out[f"{prefix}_num_items"] = tokens.apply(len).astype(np.float32)
                out[f"{prefix}_num_unique"] = unique_tokens.apply(len).astype(np.float32)
                if self.config.enable_none_features:
                    out[f"{prefix}_none_conflict"] = none_conflict.astype(np.float32)

        # ---- numeric: median impute + is_missing ----
        for col in self.numeric_cols:
            prefix = self._prefix.get(col, _snake_case_name(col))
            if col == "completed_semester":
                cleaned, outlier = self._clean_completed_semester(df[col])
                miss = cleaned.isna()
                values = cleaned.fillna(self.numeric_medians.get(col, 0.0)).astype(np.float32)
                out[prefix] = values
                out[f"is_missing_{prefix}"] = miss.astype(np.float32)
                out["completed_semester_is_outlier"] = outlier.astype(np.float32)
            else:
                miss = self._is_missing(df[col])
                values = pd.to_numeric(df[col], errors="coerce")
                values = values.where(~miss, np.nan).fillna(self.numeric_medians.get(col, 0.0)).astype(np.float32)
                out[prefix] = values
                out[f"is_missing_{prefix}"] = miss.astype(np.float32)

        # ---- categorical: freq/log_freq/is_rare (+ optional top-k one-hot) ----
        for col in self.categorical_cols:
            prefix = self._prefix.get(col, _snake_case_name(col))
            s = self._normalize_categorical(df[col])

            miss = self._is_missing(df[col])
            out[f"is_missing_{prefix}"] = miss.astype(np.float32)
            out[f"is_answered_{prefix}"] = (~miss).astype(np.float32)
            if self.config.enable_none_features:
                out[f"is_none_{prefix}"] = self._is_none(df[col]).astype(np.float32)

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
        multilabel_cache: dict[str, pd.Series] = {}
        multilabel_none_cache: dict[str, pd.Series] = {}
        for col in self.multilabel_cols:
            prefix = self._prefix.get(col, _snake_case_name(col))
            tokens, unique_tokens, none_flag, none_conflict = self._split_multilabel(col, df[col])
            miss = self._is_missing(df[col])
            out[f"is_missing_{prefix}"] = miss.astype(np.float32)
            out[f"is_answered_{prefix}"] = (~miss).astype(np.float32)
            if self.config.enable_none_features:
                out[f"is_none_{prefix}"] = none_flag.astype(np.float32)
                out[f"is_none_conflict_{prefix}"] = none_conflict.astype(np.float32)

            out[f"{prefix}_has_any"] = unique_tokens.apply(lambda xs: float(len(xs) > 0)).astype(np.float32)
            out[f"{prefix}_num_items"] = tokens.apply(len).astype(np.float32)
            out[f"{prefix}_num_unique"] = unique_tokens.apply(len).astype(np.float32)

            multilabel_cache[col] = unique_tokens
            multilabel_none_cache[col] = none_flag

            top_tokens = self.multilabel_top_tokens.get(col) or []
            if top_tokens:
                def as_set(items: list[str]) -> set[str]:
                    return set(items)

                sets = unique_tokens.apply(as_set)
                for i, tok in enumerate(top_tokens):
                    out[f"{prefix}_tok_{i}"] = sets.apply(lambda s: float(tok in s)).astype(np.float32)

            # multilabel svd embedding
            model = self._multilabel_svd_models.get(col)
            if model is not None:
                vectorizer, svd = model
                docs = unique_tokens.apply(lambda xs: " ".join(xs))
                X = vectorizer.transform(docs)
                emb = svd.transform(X).astype(np.float32)
                for j in range(emb.shape[1]):
                    out[f"{prefix}_ml_svd_{j}"] = emb[:, j]

        # ---- certificate slot features ----
        if self.config.enable_certificate_slots:
            slots = self.config.certificate_slots or {}
            slot_patterns: dict[str, list[str]] = {}
            for slot, patterns in slots.items():
                normed = []
                for p in patterns or []:
                    p_norm = self._normalize_text(p).replace(" ", "").lower()
                    if p_norm:
                        normed.append(p_norm)
                if normed:
                    slot_patterns[str(slot)] = normed

            def extract_slots(tokens: list[str]) -> set[str]:
                found: set[str] = set()
                for tok in tokens or []:
                    tok_norm = self._normalize_text(tok).replace(" ", "").lower()
                    if not tok_norm:
                        continue
                    for slot, patterns in slot_patterns.items():
                        if slot in found:
                            continue
                        for pat in patterns:
                            if pat and pat in tok_norm:
                                found.add(slot)
                                break
                return found

            acq_tokens = multilabel_cache.get("certificate_acquisition")
            des_tokens = multilabel_cache.get("desired_certificate")
            acq_sets = [set() for _ in range(len(df))]
            des_sets = [set() for _ in range(len(df))]
            if acq_tokens is not None:
                acq_sets = [extract_slots(xs) for xs in acq_tokens]
            if des_tokens is not None:
                des_sets = [extract_slots(xs) for xs in des_tokens]

            for slot in slot_patterns.keys():
                name = _snake_case_name(slot)
                out[f"acq_has_{name}"] = np.array([float(slot in s) for s in acq_sets], dtype=np.float32)
                out[f"des_has_{name}"] = np.array([float(slot in s) for s in des_sets], dtype=np.float32)

            out["acq_cnt"] = np.array([len(s) for s in acq_sets], dtype=np.float32)
            out["des_cnt"] = np.array([len(s) for s in des_sets], dtype=np.float32)
            out["overlap_cnt"] = np.array([len(a & d) for a, d in zip(acq_sets, des_sets)], dtype=np.float32)
            out["need_cnt"] = np.array([len(d - a) for a, d in zip(acq_sets, des_sets)], dtype=np.float32)

        # ---- major mapping flags ----
        if major_map_failed:
            out["major1_1_map_failed"] = major_map_failed.get("major1_1_coarse", pd.Series(0.0, index=df.index)).to_numpy()
            out["major1_2_map_failed"] = major_map_failed.get("major1_2_coarse", pd.Series(0.0, index=df.index)).to_numpy()
            out["major_map_failed"] = np.maximum(
                np.asarray(out["major1_1_map_failed"], dtype=np.float32),
                np.asarray(out["major1_2_map_failed"], dtype=np.float32),
            )
            out["major1_1_map_ambiguous"] = major_map_ambiguous.get("major1_1_coarse", pd.Series(0.0, index=df.index)).to_numpy()
            out["major1_2_map_ambiguous"] = major_map_ambiguous.get("major1_2_coarse", pd.Series(0.0, index=df.index)).to_numpy()
            out["major_map_ambiguous"] = np.maximum(
                np.asarray(out["major1_1_map_ambiguous"], dtype=np.float32),
                np.asarray(out["major1_2_map_ambiguous"], dtype=np.float32),
            )

        # ---- text: T0/T1/T2 ----
        text_mode = (self.config.text_mode or "T1").upper()
        for col in self.text_cols:
            prefix = self._prefix.get(col, _snake_case_name(col))
            s = df[col].astype("string").fillna("").apply(self._normalize_text).astype(str)
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
        if self._cluster_cols and self._kmeans_by_k:
            X_miss = np.column_stack([self._is_missing(df[c]).astype(np.float32).to_numpy() for c in self._cluster_cols])
            for k, kmeans in sorted(self._kmeans_by_k.items()):
                cluster_id = kmeans.predict(X_miss).astype(np.int32)
                dist = kmeans.transform(X_miss).astype(np.float32)
                if self._primary_cluster_k is not None and int(k) == int(self._primary_cluster_k):
                    out["missing_cluster_id"] = cluster_id.astype(np.float32)
                    for j in range(dist.shape[1]):
                        out[f"missing_cluster_dist_{j}"] = dist[:, j]
                    for j in range(dist.shape[1]):
                        out[f"missing_cluster_oh_{j}"] = (cluster_id == j).astype(np.float32)
                else:
                    out[f"missing_cluster_k{k}_id"] = cluster_id.astype(np.float32)
                    for j in range(dist.shape[1]):
                        out[f"missing_cluster_k{k}_dist_{j}"] = dist[:, j]
                    for j in range(dist.shape[1]):
                        out[f"missing_cluster_k{k}_oh_{j}"] = (cluster_id == j).astype(np.float32)

            if self._gmm is not None:
                try:
                    prob = self._gmm.predict_proba(X_miss).astype(np.float32)
                    out["missing_gmm_id"] = np.argmax(prob, axis=1).astype(np.float32)
                    for j in range(prob.shape[1]):
                        out[f"missing_gmm_prob_{j}"] = prob[:, j]
                except Exception:
                    pass

        processed = pd.DataFrame(out)

        if (
            self.config.enable_domain_adaptation_feature
            and self._domain_clf is not None
            and self._domain_scaler is not None
            and self._domain_feature_cols
        ):
            try:
                feature_cols = list(self._domain_feature_cols)
                X = processed.reindex(columns=feature_cols, fill_value=0.0).to_numpy(dtype=np.float32, copy=True)
                X_scaled = self._domain_scaler.transform(X)
                proba = self._domain_clf.predict_proba(X_scaled)[:, 1].astype(np.float32)
                col_name = str(self.config.domain_adaptation_feature_name or "p_test").strip() or "p_test"
                processed[col_name] = proba
            except Exception:
                pass

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
        df_unlabeled = None
        if config is not None and (
            getattr(config, "enable_transductive_fit", False) or getattr(config, "enable_domain_adaptation_feature", False)
        ):
            transductive_path = str(getattr(config, "transductive_fit_csv", "") or "").strip()
            if transductive_path and transductive_path != str(file_path):
                try:
                    df_unlabeled = pd.read_csv(transductive_path)
                except Exception:
                    df_unlabeled = None

        fitted.fit(df, df_unlabeled=df_unlabeled)
        processed = fitted.transform(df, keep_target=True)

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
