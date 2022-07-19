"""Вспомогательные функции"""

import pickle
import unicodedata
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


def decode_text(raw, encode='utf-8', errors='ignore', form='NFKD'):
    """
    Декодирование текста
    """
    nfkd_form = unicodedata.normalize(form, raw)
    result = nfkd_form.encode(encode, errors).decode(encode)
    return result


def get_datetime_features(data: pd.DataFrame, timestamp_col: str='publish_date') -> pd.DataFrame:
    """Создание фич даты времени

    Args:
        data (pd.DataFrame): _description_
        timestamp_col (str, optional): _description_. Defaults to 'timestamp'.

    Returns:
        pd.DataFrame: _description_
    """

    timestamp = data[timestamp_col].copy()
    if not is_datetime64_any_dtype(timestamp):
        timestamp = pd.to_datetime(timestamp)

    result = pd.DataFrame(dtype=object, index=data.index)

    result['hour'] = timestamp.dt.hour
    result['day'] = timestamp.dt.day
    result['weekday'] = timestamp.dt.weekday
    # result['month'] = timestamp.dt.month

    return result


def get_unique_categorical(data, prefix=''):
    """Функция для анализа - смотрит, сколько в датасете уникальных категориальных признаков

    Args:
        data (_type_): _description_
        prefix (str, optional): _description_. Defaults to ''.

    Returns:
        _type_: _description_
    """

    unique_sessions = set(data['session'])
    unique_sessions_frac = len(unique_sessions) / data.shape[0]

    exploded_authors = data['authors'].explode()
    unique_authors = set(exploded_authors)
    unique_authors_frac = len(unique_authors) / exploded_authors.shape[0]

    exploded_tags = data['tags'].explode()
    unique_tags = set(exploded_tags)
    unique_tags_frac = len(unique_tags) / exploded_tags.shape[0]

    for uniq, uniq_name in zip(
        [unique_sessions_frac, unique_authors_frac, unique_tags_frac],
        ['unique_sessions_frac', 'unique_authors_frac', 'unique_tags_frac']
    ):
        print(f'{prefix}{uniq_name} {uniq:.2%}')
    return unique_sessions, unique_authors, unique_tags


def get_datasets_cat_intersection(train, test):
    """Функция для анализа - смотрит, сколько пересекающихся значений
    категориальных признаков между датасетами

    Args:
        train (_type_): _description_
        test (_type_): _description_

    Returns:
        _type_: _description_
    """
    unique_sessions_train, unique_authors_train, unique_tags_train = \
        get_unique_categorical(train, prefix='train ')
    print()
    unique_sessions_test, unique_authors_test, unique_tags_test = \
        get_unique_categorical(test, prefix='test ')
    print()
    same_sessions = unique_sessions_train & unique_sessions_test
    same_authors = unique_authors_train & unique_authors_test
    same_tags = unique_tags_train & unique_tags_test

    for same, same_name, test_uniq in zip(
        [same_sessions, same_authors, same_tags],
        ['same_sessions', 'same_authors', 'same_tags'],
        [unique_sessions_test, unique_authors_test, unique_tags_test]):
        print(f'{same_name} {len(same) / len(test_uniq):.2%}')

    return same_sessions, same_authors, same_tags


def my_dropna(data: pd.DataFrame, notna_ratio: float=.05, axis: int=1) -> pd.DataFrame:
    """dropna в более удобной форме для выбора доли нанов

    Args:
        data (pd.DataFrame): _description_
        notna_ratio (float, optional): _description_. Defaults to .05.
        axis (int, optional): _description_. Defaults to 1.

    Returns:
        pd.DataFrame: _description_
    """
    n_nan = notna_ratio * data.shape[0]
    return data.dropna(axis=axis, thresh=n_nan)


def load_pickle(path: str) -> None:
    """Загрузка пикла

    Args:
        path (str): _description_

    Returns:
        _type_: _description_
    """
    with open(path, 'rb') as file:
        return pickle.load(file)


def dump_pickle(obj, path: str) -> None:
    """Дамп пикла

    Args:
        obj (_type_): _description_
        path (str): _description_
    """
    with open(path, 'wb') as file:
        pickle.dump(obj, file)
