from pathlib import Path

import pandas as pd


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Загружает данные из указанной директории
    Args:
        data_dir (Path): директория с "сырыми" данными

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train_data, test_data, categories_data

    """
    categories_data = pd.read_csv(data_dir / 'categories_tree.csv')
    test_data = pd.read_parquet(data_dir / 'test.parquet')
    train_data = pd.read_parquet(data_dir / 'train.parquet')

    return train_data, test_data, categories_data
