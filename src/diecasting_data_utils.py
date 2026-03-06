import pandas as pd
from pathlib import Path

# 프로젝트 루트 기준 raw 데이터 경로
_ORIGIN_DATA_DIR = Path(__file__).parent.parent / "data"
# 프로젝트 루트 기준 전처리 완료 데이터 경로
_PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data_processed"

def load_origin_data() -> pd.DataFrame:
    """
    data/DieCasting_Quality_Raw_Data.csv 를 로드해서 반환.

    Returns:
        멀티헤더 DataFrame
    """
    path = _ORIGIN_DATA_DIR / "DieCasting_Quality_Raw_Data.csv"
    return pd.read_csv(path, header=[0, 1])

def load_processed_data(product_type: int) -> pd.DataFrame:
    """
    data_processed/product_type_{product_type}.csv 를 로드해서 반환.

    Args:
        product_type: 1 또는 2

    Returns:
        멀티헤더 DataFrame
    """
    if product_type not in [1, 2]:
        raise Exception("product_type 파라미터가 필요합니다.  ex)1 or 2")
    path = _PROCESSED_DATA_DIR / f"product_type_{product_type}.csv"
    return pd.read_csv(path, header=[0, 1])

def remove_columns(product_type, category_name, colum_name):
    if product_type not in [1, 2]:
        raise Exception(f"존재하지 않는 product_type: {product_type}")
    data = load_processed_data(product_type)
    col = (category_name, colum_name)
    if col not in data.columns:
        raise Exception(f"존재하지 않는 컬럼: {col}")
    return data.drop(columns=[col])
