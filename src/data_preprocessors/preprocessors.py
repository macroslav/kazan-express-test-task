import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def fill_description_nans(data: pd.DataFrame,
                          fill_value: str = 'описания нет') -> pd.DataFrame:
    """
    Заполняет пропуски и пустые описания
    Args:
        fill_value (str): значение для заполнения пропусков в описаниях
        data (pd.DataFrame): тренировочный/тестовый DataFrame

    Returns:
        pd.DataFrame: данные с заполненными пропусками

    """
    data['short_description'] = data['short_description'].fillna(fill_value)
    data.loc[data['short_description'] == '', 'short_description'] = fill_value
    data.loc[data['short_description'] == 'None', 'short_description'] = fill_value

    return data


def preprocess_data(data: pd.DataFrame,
                    drop_columns: list[str]) -> pd.DataFrame:
    """
    Возвращает DataFrame с обработанными признаками
    Args:
        drop_columns: колонки, которые необходимо удалить из данных
        data: исходные данные с необработанными признаками

    Returns:
        DataFrame с обработанными признаками
    """
    data['title'] = data['title'].str.lower()
    data['short_description'] = data['short_description'].str.lower()
    data['description'] = data['title'] + ' ' + data['short_description']

    data = data.drop(columns=drop_columns)
    return data


def get_categories_to_change(data: pd.DataFrame, threshold: int = 100) -> list[int]:
    """
    Возвращает список категорий, которые необходимо заменить на родительскую категорию
    Args:
        data: train данные, в которых необходимо заменить категории на parent категории
        threshold: порог для замены, если в категории количество уникальных товаров ниже, чем данный порог, то категория
        заменяется на родительскую

    Returns:
        list[int]: список категорий, которые необходимо заменить на родительскую

    """
    products_per_categories = data.groupby('new_category', as_index=False) \
        .agg(unique_products=('id', 'count')).sort_values(by='unique_products')

    return products_per_categories[products_per_categories.unique_products < threshold].new_category.tolist()


def change_category_to_parent(categories_to_change: list[int],
                              data: pd.DataFrame,
                              categories_data: pd.DataFrame) -> pd.DataFrame:
    """
    Заменяет категории из списка на их родительские категории
    Args:
        categories_to_change: список категорий для замены
        data: train данные
        categories_data: pd.DataFrame с данными о категориях

    Returns:
        pd.DataFrame с замененными категориями
    """
    category_to_parent_dict = dict(
        categories_data.loc[categories_data.id.isin(categories_to_change), ['id', 'parent_id']].values)
    data.loc[data.new_category.isin(categories_to_change), 'new_category'] = data.new_category.map(
        category_to_parent_dict)
    data['new_category'] = data['new_category'].astype(int)

    return data


def build_full_path_str(category_id: int,
                        categories_data: pd.DataFrame) -> str:
    """
    Возвращает строку с полным деревом категорий
    Args:
        categories_data: pd.DataFrame с данными о категориях
        category_id: категория, для которой необходимо построить дерево

    Returns:
        Строка с полным деревом родителей данной категории
    """
    labels_path = _build_full_path(category_id, type_='str', categories_data=categories_data)

    return " ".join(labels_path[::-1])


def build_full_path_id(category_id: int,
                       categories_data: pd.DataFrame) -> list[int]:
    """
    Возвращает полное дерево категорий для указанной категории
    Args:
        category_id: категория, для которой необходимо построить дерево
        categories_data: pd.DataFrame с данными о категориях

    Returns:
        Список родителей данной категории
    """
    ids_path = _build_full_path(category_id, type_='int', categories_data=categories_data)
    return ids_path


def _build_full_path(category_id: int, categories_data: pd.DataFrame, type_: str = 'int') -> list:
    """
    Строит дерево названий и id для выбранной категории
    Args:
        category_id: категория, для которой необходимо построить дерево
        categories_data: pd.DataFrame с данными о категориях
        type_: тип дерева str или int

    Returns:
        Возвращает дерево выбранного типа
    """
    labels_path = []
    id_path = []
    leaf_category = category_id

    exit_flag = False
    while not exit_flag:
        try:
            parent = categories_data.loc[categories_data.id == leaf_category, 'parent_id'].values[0]
        except IndexError:
            exit_flag = True
            continue

        label = categories_data.loc[categories_data.id == leaf_category, 'title'].values[0]
        labels_path.append(label)
        id_path.append(leaf_category)

        if parent == 0 or parent == 1:
            exit_flag = True
        else:
            leaf_category = parent
    if type_ == 'str':
        return labels_path
    elif type_ == 'int':
        return id_path


def process_categories_data(categories_data: pd.DataFrame, used_categories: list[int]) -> pd.DataFrame:
    """
    Обрабатывает данные о категориях
    Args:
        categories_data: данные о категориях
        used_categories: категории, которые присутствуют в тренировочном датасете

    Returns:
        Обработанные данные о категориях
    """
    categories_data['full_path_id'] = ''
    categories_data['full_path_id'] = categories_data['full_path_id'].astype(object)
    for row_index in categories_data.index:
        cat_id = categories_data.loc[row_index, 'id']
        if cat_id in used_categories:
            full_path_id = build_full_path_id(cat_id, categories_data)
            categories_data.at[row_index, 'full_path_id'] = full_path_id
    return categories_data


def extract_text_features(train_data: pd.DataFrame,
                          test_data: pd.DataFrame,
                          max_features: int = 5000) -> tuple:
    """
    Извлекает фичи из описаний товаров при помощи CountVectorizer
    Args:
        train_data: тренировочный датасет
        test_data: тестовый датасет
        max_features: максимальное количество признаков из CountVectorizer

    Returns:
        Кортеж с train_features и test_features
    """
    vectorizer = CountVectorizer(ngram_range=(1, 1), binary=True, max_features=max_features)
    vectorizer.fit(train_data.description.values.tolist() + test_data.description.values.tolist())

    train_features = vectorizer.transform(train_data.description.values)
    test_features = vectorizer.transform(test_data.description.values)

    return train_features, test_features
