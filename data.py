'''

데이터 전처리용

TODO : 데이터 전처리 함수 작성. 데이터 전처리 목적이 아닌 함수는 utils.py에 작성 후 import 하여 여기에서 호출할것

데이터 반환 양식은 pandas DataFrame으로 통일한다.

'''
import pandas as pd
import numpy as np


def preprocess_data(file_path, is_train=True, train_cols=None, id_label="ID", target_label="completed"):
    df = pd.read_csv(file_path)
    missing_cols = [c for c in [id_label, target_label] if is_train and c not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing from training data: {missing_cols}")

    ids = df[id_label] if id_label in df.columns else None
    y = df[target_label] if is_train else None
    
    # 1. Sparsity Control & Column Drop
    if is_train:
        null_threshold = 0.9
        # 결측률 너무 높은 컬럼 제거 (ID/Target은 유지)
        drop_cols = df.columns[df.isnull().sum() / len(df) > null_threshold].tolist()
        drop_cols = [c for c in drop_cols if c not in [id_label, target_label]]
    else:
        # 테스트는 학습 컬럼과 정확히 맞추되 target은 제외
        if train_cols is None:
            raise ValueError("train_cols is required when is_train=False")
        train_cols_no_target = [c for c in train_cols if c != target_label]
        # 누락 컬럼은 NaN으로 생성되어 이후 처리에서 보정
        df_clean = df.reindex(columns=train_cols_no_target).copy()
        drop_cols = []
    
    if is_train:
        df_clean = df.drop(columns=drop_cols)
    
    # 2. Dynamic Feature Assignment (에러 원인 원천 차단)
    # 텍스트 피처 후보군 (서술형 문항)
    potential_text = ['whyBDA', 'what_to_gain', 'incumbents_lecture_scale_reason', 'onedayclass_topic']
    text_features = [c for c in potential_text if c in df_clean.columns and c not in [id_label, target_label]]
    
    # 나머지 문자열/객체 타입은 모두 범주형으로 간주
    cat_features = df_clean.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    cat_features = [c for c in cat_features if c not in text_features and c not in [id_label, target_label]]
    
    # 3. Handling String Noise in Numerical Columns
    # 수치형이어야 하는데 '해당없음' 등이 섞인 경우를 대비해 강제 형변환
    num_features = df_clean.select_dtypes(exclude=['object', 'category', 'bool']).columns.tolist()
    num_features = [c for c in num_features if c not in [id_label, target_label]]
    for col in num_features:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
    # 범주형/텍스트 데이터의 결측치 처리
    for col in cat_features + text_features:
        df_clean[col] = df_clean[col].astype(str).replace(['nan', 'None', 'None '], 'Unknown')

    # 4. Encode non-numeric features as numeric (CatBoost cat_features not passed in train)
    # Use a deterministic hash to keep train/test consistent without storing mappings.
    non_numeric_cols = df_clean.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    non_numeric_cols = [c for c in non_numeric_cols if c not in [id_label, target_label]]
    for col in non_numeric_cols:
        hashed = pd.util.hash_pandas_object(df_clean[col].astype(str), index=False)
        df_clean[col] = (hashed % 1_000_000).astype(np.float32)
        
    if is_train:
        return df_clean, y, cat_features, text_features
    else:
        return df_clean, ids, cat_features, text_features
