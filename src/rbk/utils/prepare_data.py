"""Функции для подготовки данных
"""
import os
import re
from typing import Optional
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from tqdm import tqdm

from rbk.utils import helpers, text_processing
from rbk.utils.global_vars import PARSING_COLS, RUS_LETTERS, \
    ECONOMIC_TRADING_NAMES, ECONOMIC_COLS_MAPPING, ECONOMIC_NUM_COLS,\
        ROOT_ABSPATH, ECONOMIC_DATA_PATH, COVID_PATH, COVID_COLS, COVID_ISO_CODE, WEATHER_PATH


def prepare_weather_df(path_to_weather: str) -> pd.DataFrame:
    """Обработка файла с данными о погоде с сайта rp5.ru

    Args:
        path_to_weather (_type_): _description_

    Returns:
        _type_: _description_
    """
    weather = pd.read_excel(path_to_weather)
    dt_col = weather.columns[0]
    rename_dict = {
        dt_col: 'datetime',
        'T': 'temperature',
        'Po': 'pressure',
        'U': 'air_humidity',
        'Ff': 'wind_speed',
        'RRR': 'rainfall'
    }
    weather = weather[rename_dict.keys()]
    weather = weather.rename(columns=rename_dict)
    weather['rainfall'] = \
        weather['rainfall'].apply(lambda x: re.sub(r'[^0-9.]', '', x) if isinstance(x, str) else x)
    weather['rainfall'] = weather['rainfall'].replace('', np.nan)
    weather['rainfall'] = weather['rainfall'].fillna(0)
    weather = weather.dropna()
    weather['datetime'] = pd.to_datetime(weather['datetime'], format='%d.%m.%Y %H:%M')
    weather['date'] = weather['datetime'].dt.date
    weather = \
        weather.groupby('date')[['temperature', 'pressure', 'air_humidity', 'wind_speed', 'rainfall']].mean()
    return weather


def get_splited_title(train: pd.DataFrame) -> pd.DataFrame:
    """Разбивает заголовок статьи по переносам

    Args:
        train (_type_): _description_

    Returns:
        _type_: _description_
    """
    splited_title = train['title'].str.split('\n').apply(pd.Series)
    for col in splited_title:
        splited_title[col] = splited_title[col].str.strip()
    splited_title = splited_title.replace({'': np.nan})
    return splited_title


def prepare_splited_title(splited_title: pd.DataFrame) -> pd.DataFrame:
    """Тайтл представляет собой смесь из заголовка, категории и даты
    Разделяем это всё

    Args:
        splited_title (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    splited_title = splited_title.copy()

    splited_title.loc[splited_title[6].isna(), 6] = \
        splited_title.loc[splited_title[6].isna(), 5]
    splited_title.loc[splited_title[1].notna(), 0] += \
        '. ' + splited_title.loc[splited_title[1].notna(), 1]

    theme_and_local_time = splited_title[6].apply(
        lambda x: [el.strip() for el in x.split(',')] if isinstance(x, str) else x
        )
    theme_and_local_time = theme_and_local_time.apply(pd.Series)

    theme_and_local_time.columns = ['theme', 'date', 'local_time']

    theme_and_local_time.loc[
        theme_and_local_time['local_time'].isna(),
        'local_time'
        ] = \
        theme_and_local_time.loc[theme_and_local_time['local_time'].isna(), 'date']

    theme_and_local_time = theme_and_local_time.drop(columns='date')

    local_time = theme_and_local_time['local_time'].str.strip().str.split(':')
    local_time = local_time.apply(pd.Series)
    local_time.columns = ['local_hour', 'local_minute']
    local_time['local_hour'] = pd.to_numeric(local_time['local_hour'])
    local_time['local_minute'] = pd.to_numeric(local_time['local_minute'])

    splited_title = splited_title.rename(columns={0: 'title'})

    return splited_title[['title']]\
        .join(theme_and_local_time[['theme']]).join(local_time[['local_hour']])


def extend_with_title_info(train: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        train (_type_): _description_

    Returns:
        _type_: _description_
    """
    train = train.copy()
    train['title'] = train['title'].apply(helpers.decode_text)
    splited_title = get_splited_title(train)
    prepared_title = prepare_splited_title(splited_title)
    result = train.drop(columns='title')
    return result.join(prepared_title)


def str_to_list(train: pd.DataFrame, column: str) -> pd.Series:
    """В данных листы превратились в строки
    Функция возвращает их к формату листов

    Args:
        train (pd.DataFrame): _description_
        column (str): _description_

    Returns:
        pd.Series: _description_
    """
    series = train[column].copy()
    series = series.str.replace('[', '', regex=False).str.replace(']', '', regex=False)
    series = series.apply(lambda x: re.sub(f'[^0-9a-zA-Z,{RUS_LETTERS}]', '', x))
    series = series.str.split(',')
    return series


def prepare_parsing_df(parsing_series: pd.Series or dict,
                       colnames: str=None,
                       postfix: str='') -> pd.DataFrame:
    """Приводит результат парсинга в красивый вид

    Args:
        parsing_df (pd.Series): _description_
        colnames (str, optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    if colnames is None:
        colnames = PARSING_COLS
    if isinstance(parsing_series, dict):
        parsing_series = pd.Series(parsing_series)
    if not isinstance(parsing_series, (dict, pd.Series)):
        raise TypeError(f'parsing_series must be pd.Series or dict, {type(parsing_series)} found')
    result = parsing_series.apply(pd.Series)
    result.columns = [f'{col}{postfix}' for col in PARSING_COLS]
    return result


def convert_str_to_num(train: pd.DataFrame,
                      views_col: str,
                      num_type: type=int) -> pd.Series:
    """Делает из строки число нужного типа

    Args:
        train (pd.DataFrame): _description_
        views_col (str): _description_
        num_type (type, optional): _description_. Defaults to int.

    Returns:
        pd.Series: _description_
    """
    result = train[views_col].apply(
        lambda x: num_type(re.sub(r'[^0-9]', '', x)) if isinstance(x, str) else x
        )
    return result


def prepare_num_format(num_text: str) -> str:
    """В экономических данных встречаются штуки вида 17.977,88
    преобразую их

    Args:
        num_text (_type_): _description_

    Returns:
        _type_: _description_
    """
    if not isinstance(num_text, str):
        return num_text
    num_text = num_text.replace('%', '')
    cleaned = re.sub(r'[^.,]', '', num_text)
    if cleaned == '.,':
        return num_text.replace('.', '').replace(',', '.')
    if cleaned == ',':
        return num_text.replace(',', '.')
    if cleaned == '' or cleaned == '.':
        return num_text
    print(f'unknown num format {num_text}, {cleaned}')
    return np.nan


def prepare_trading(economic_folder: str,
                    trading_names: Optional[list]=None,
                    trading_cols_mapping: Optional[dict]=None,
                    num_cols: Optional[list[str]]=None) -> pd.DataFrame:
    """Объединяет файлы с данными о торгующихся индексах, валютах, нефти и проч
    В одну плоскую таблицу

    Args:
        economic_folder (str): _description_
        trading_names (Optional[list], optional): _description_. Defaults to None.
        trading_cols_mapping (Optional[dict], optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    if trading_names is None:
        trading_names = ECONOMIC_TRADING_NAMES
    if trading_cols_mapping is None:
        trading_cols_mapping = ECONOMIC_COLS_MAPPING
    if num_cols is None:
        num_cols = ECONOMIC_NUM_COLS

    tradings_dfs_list = []
    for name in trading_names:
        df = pd.read_excel(os.path.join(economic_folder, f'{name}.xlsx'), dtype=object)
        for col in num_cols:
            try:
                df[col] = df[col].apply(prepare_num_format)
                df[col] = pd.to_numeric(df[col])
            except KeyError:
                print(f"{col} not found in {name} when pd.to_numeric")

        df['type'] = name
        tradings_dfs_list.append(df)

    result = pd.concat(tradings_dfs_list).reset_index(drop=True)
    result = result.rename(columns=trading_cols_mapping)

    return result.sort_values(by=['type', 'date'])


def calc_object_trading_features(tradings_df: pd.DataFrame,
                                   trading_obj: str,
                                   window: int=7) -> pd.DataFrame:
    """
    Args:
        tradings_df (pd.DataFrame): _description_
        trading_obj (str): _description_
        window (int, optional): _description_. Defaults to 7.

    Returns:
        _type_: _description_
    """

    cols_to_drop = [
        'price_open',
        'max_price',
        'min_price',
        'volume',
        'growth_perc',
        f'price_mean_{window}d',
        'type'
        ]

    obj_condition = tradings_df['type'] == trading_obj
    tradings_df = tradings_df[obj_condition].copy()

    price_shift = tradings_df['price'].shift()
    tradings_df['grows'] = (tradings_df['price'] - price_shift) / price_shift

    tradings_df[f'price_mean_{window}d'] = \
        tradings_df['price'].rolling(window=window, center=False).mean()
    tradings_df[f'price_std_{window}d'] = \
        tradings_df['price'].rolling(window=window, center=False).std() / tradings_df[f'price_mean_{window}d']

    tradings_df[f'grows_{window}d'] = \
        (tradings_df['price'] - tradings_df[f'price_mean_{window}d']) / tradings_df[f'price_mean_{window}d']
    tradings_df = tradings_df.drop(columns=cols_to_drop)
    tradings_df.columns = [f'{trading_obj}_{col}' if col != 'date' else col for col in tradings_df.columns]
    tradings_df.columns = [col.replace('_price', '') for col in tradings_df.columns]
    return tradings_df


def extend_by_trading_features(train: pd.DataFrame,
                               economic_folder: str=None,
                               trading_names: Optional[list]=None,
                               window: int=7) -> pd.DataFrame:
    """Докидывает в данные информацию об экономических показателях

    Args:
        train (pd.DataFrame): _description_
        economic_folder (str, optional): _description_. Defaults to None.
        trading_names (Optional[list], optional): _description_. Defaults to None.
        window (int, optional): _description_. Defaults to 7.

    Returns:
        pd.DataFrame: _description_
    """
    train = train.copy()
    if economic_folder is None:
        economic_folder = os.path.join(ROOT_ABSPATH, ECONOMIC_DATA_PATH)
    if trading_names is None:
        trading_names = ECONOMIC_TRADING_NAMES

    tradings_df = prepare_trading(economic_folder)
    df_date = train['publish_date'].dt.date.reset_index().set_index('publish_date')

    for trading_obj in ECONOMIC_TRADING_NAMES:
        trading_features = calc_object_trading_features(tradings_df, trading_obj, window=window)
        trading_features.index = trading_features['date'].dt.date

        result = df_date.join(trading_features, how='left').sort_index()
        result = result.set_index('document_id').drop(columns='date')

        result[trading_obj] = result[trading_obj].fillna(method='ffill')
        result = result.dropna(how='all')
        result = result.fillna(0)

        train = train.join(result, how='left')
    return train


def extend_by_inflation_and_key_rate(train: pd.DataFrame,
                                    economic_folder: str=None,
                                    filename: str='br_inflation_key_rate.xlsx') -> pd.DataFrame:
    """Докидывает в данные информацию об инфляции и ключевой ставке ЦБ

    Args:
        train (pd.DataFrame): _description_
        economic_folder (str, optional): _description_. Defaults to None.
        filename (Optional[list], optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    train = train.copy()
    if economic_folder is None:
        economic_folder = os.path.join(ROOT_ABSPATH, ECONOMIC_DATA_PATH)

    bank_data = pd.read_excel(os.path.join(economic_folder, filename), dtype=object)
    bank_data.columns = ['date', 'key_rate', 'inflation', 'target_inflation']
    bank_data['date'] = bank_data['date'].apply(

        lambda x: f"{int(x.split('.')[0])}.{int(x.split('.')[1])}"

        )

    month_year = train['publish_date'].dt.month.astype(str) + '.' + train['publish_date'].dt.year.astype(str)
    month_year.name = 'date'
    month_year = month_year.reset_index()
    result = month_year.merge(bank_data, on='date', how='left')
    result = result.drop(columns=['target_inflation', 'date']).set_index('document_id')

    for col in ['key_rate', 'inflation']:
        result[col] = pd.to_numeric(result[col])

    return train.join(result, how='left')


def extend_by_covid_features(train: pd.DataFrame,
                            covid_folder: str=None,
                            filename: str='owid-covid-data.csv',
                            covid_cols: Optional[list[str]]=None,
                            covid_iso_codes: Optional[list[str]]=None) -> pd.DataFrame:
    """Дополняет датасет данными о заболеваемоси covid

    Args:
        train (pd.DataFrame): _description_
        covid_folder (str, optional): _description_. Defaults to None.
        filename (str, optional): _description_. Defaults to 'owid-covid-train.csv'.
        covid_cols (Optional[list[str]], optional): _description_. Defaults to None.
        covid_iso_codes (Optional[list[str]], optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """

    train = train.copy()
    if covid_folder is None:
        covid_folder = os.path.join(ROOT_ABSPATH, COVID_PATH)
    if covid_iso_codes is None:
        covid_iso_codes = COVID_ISO_CODE
    if covid_cols is None:
        covid_cols = COVID_COLS

    df_date = train['publish_date'].dt.date.reset_index().set_index('publish_date')

    covid_filepath = os.path.join(covid_folder, filename)
    covid_data = pd.read_csv(covid_filepath)
    covid_data['date'] = pd.to_datetime(covid_data['date']).dt.date

    result = df_date.copy()

    for code in covid_iso_codes:
        covid_local = covid_data.loc[covid_data['iso_code']==code, covid_cols]
        covid_local['date'] = pd.to_datetime(covid_local['date']).dt.date
        covid_local = covid_local.set_index('date')
        covid_local.columns = [f'{col}_{code}' for col in covid_local.columns]
        result = result.join(covid_local, how='left')
    result = result.set_index('document_id')
    return train.join(result, how='left')


def extend_by_wheather(train: pd.DataFrame,
                        weather_folder: str=None,
                        filename: str='moscow.xls') -> pd.DataFrame:
    """Дополняет датасет данными о погоде

    Args:
        train (pd.DataFrame): _description_
        weather_folder (str, optional): _description_. Defaults to None.
        filename (str, optional): _description_. Defaults to 'moscow.xls'.

    Returns:
        pd.DataFrame: _description_
    """
    if weather_folder is None:
        weather_folder = os.path.join(ROOT_ABSPATH, WEATHER_PATH)
    weather_path = os.path.join(weather_folder, filename)
    weather = prepare_weather_df(weather_path)
    df_date = train['publish_date'].dt.date
    df_date = df_date.reset_index().set_index('publish_date')
    joined = df_date.join(weather, how='left').set_index('document_id')
    return train.join(joined)


def get_stats_for_categorical(
    article_row,
    train: pd.DataFrame,
    offset: int,
    dt_col: str='publish_date',
    categorical_col: str='tags') -> list:
    """Получает статистику для категориального признака из строчки о статье
    сколько раз в статьях за offset до date встречались такие же категории
    например, сколько статей с тегами из статьи
    или сколько статей от авторов, которые написали статью

    Args:
        article_row (_type_): _description_
        train (pd.DataFrame): _description_
        offset (int): _description_
        dt_col (str, optional): _description_. Defaults to 'publish_date'.
        categorical_col (str, optional): _description_. Defaults to 'tags'.

    Raises:
        ValueError: _description_

    Returns:
        list: _description_
    """
    target_vals = article_row[categorical_col]

    if not isinstance(target_vals, (list, str)):
        raise ValueError(f'target_vals must be list or str, {type(target_vals)} found')

    if isinstance(target_vals, str):
        target_vals = {target_vals}
    else:
        target_vals = set(target_vals)

    train = train[[categorical_col, dt_col]].copy()
    end = article_row[dt_col]

    start = end + pd.to_timedelta(f'{offset}D')

    if end < start:
        start, end = end, start

    slice_cond = (train[dt_col] < end) & (train[dt_col] > start)
    # сколько всего объектов категории встретилось для статей за предыдущий период
    # например, сколько всего тегов или сколько всего авторов
    # аналог количества слов в тексте
    dt_slice = train.loc[slice_cond, categorical_col].explode(categorical_col)
    slice_size = dt_slice.shape[0] - len(target_vals)
    # если объект входит в объекты категории рассмаориваемой статьи
    # т.е. если такой автор есть в рассматриваемой статье, или тег
    intersect_size = dt_slice.isin(target_vals).sum() - len(target_vals)
    # получается что-то вроде как часто теги из статьи встречались среди тегов статей за прошлую неделю
    # аналог term frequency, где term - это, как вариант, не один therm, а несколько тегов

    if slice_size == 0:
        intersect_frac = np.nan
    else:
        intersect_frac = intersect_size / slice_size
    # сколько всего тегов или авторов встретилось в статьях за период = offset
    # и какой процент из них совпадает хотя бы с одним из тегов/авторов рассматриваемой статьи
    return slice_size, intersect_frac


def get_stats_for_all_categories(
    article_row,
    train: pd.DataFrame,
    offset: int,
    dt_col: str='publish_date',
    categorical_col: str='category') -> list:
    """Получает для всех доступных значений категории новости
    как часто какая категория встречалась

    Args:
        article_row (_type_): _description_
        train (pd.DataFrame): _description_
        offset (int): _description_
        dt_col (str, optional): _description_. Defaults to 'publish_date'.
        categorical_col (str, optional): _description_. Defaults to 'tags'.

    Raises:
        ValueError: _description_

    Returns:
        list: _description_
    """
    target_vals = article_row[categorical_col]
    ohe_encoded = pd.get_dummies(train[categorical_col])

    if not isinstance(target_vals, (list, str)):
        raise ValueError(f'target_vals must be list or str, {type(target_vals)} found')

    if isinstance(target_vals, str):
        target_vals = {target_vals}
    else:
        target_vals = set(target_vals)

    train = train[[categorical_col, dt_col]].copy()
    end = article_row[dt_col]

    start = end + pd.to_timedelta(f'{offset}D')

    if end < start:
        start, end = end, start

    slice_cond = (train[dt_col] < end) & (train[dt_col] > start)

    aggregated = ohe_encoded[slice_cond].sum()
    summary = aggregated.sum()
    aggregated = aggregated / summary
    postfix = 'last'
    if offset > 0:
        postfix = 'next'
    aggregated.index = [f'{ind}_frac_{postfix}_{abs(offset)}D' for ind in aggregated.index]
    aggregated[f'articles_count_{postfix}_{abs(offset)}D'] = summary
    return aggregated


def get_url_and_sessions_count(train: pd.DataFrame) -> pd.DataFrame:
    """Считает количество вхождений для url и сессий

    Args:
        train (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    doc_id_df = train['session'].reset_index()
    doc_id_df['url_id'] = doc_id_df.apply(lambda x: x['document_id'].replace(x['session'], ''), axis=1)
    url_count = doc_id_df.groupby('url_id')['document_id'].count()
    session_count = doc_id_df.groupby('session')['document_id'].count()
    url_count.name = 'url_id_count'
    session_count.name = 'session_count'
    url_count = url_count.reset_index()
    session_count = session_count.reset_index()
    doc_id_df = doc_id_df.merge(url_count, how='left').merge(session_count, how='left')
    return doc_id_df.set_index('document_id').drop(columns=['session'])


def preprocess_extend_raw_data(train: pd.DataFrame,
                                parsed_data: pd.DataFrame,
                                get_dt_features_from: str='publish_date',
                                # collecting_date принял равным дате получения последней новости
                                collecting_date: str='2022-05-29 23:59:59',
                                parsing_date: str='2022-06-28 15:18:00') -> pd.DataFrame:
    """Собирает датасет со всеми признаками

    Args:
        train (pd.DataFrame): _description_
        parsed_data (pd.DataFrame): _description_
        collecting_date (_type_, optional): _description_. Defaults to '2022-05-29 23:59:59'.
        parsing_date (_type_, optional): _description_. Defaults to '2022-06-28 15:18:00'.

    Returns:
        pd.DataFrame: _description_
    """
    train = train.copy()
    train = extend_with_title_info(train)
    train['authors'] = str_to_list(train, 'authors')
    train['tags'] = str_to_list(train, 'tags')

    parsing_df = prepare_parsing_df(parsed_data, postfix='_parsed')
    train = train.join(parsing_df)
    train['publish_date'] = pd.to_datetime(train['publish_date'])
    train['publish_date_parsed'] = pd.to_datetime(train['datetime_parsed']).dt.tz_convert(None)
    train['views_parsed'] = convert_str_to_num(train, 'views_parsed')
    train = train.join(helpers.get_datetime_features(train, timestamp_col=get_dt_features_from))

    train['days_from_collecting'] = (
        pd.to_datetime(collecting_date) - train['publish_date']
        ).dt.days + 1
    train['days_from_parsing'] = (
        pd.to_datetime(parsing_date) - train['publish_date']
        ).dt.days + 1

    for col in train.columns:
        if is_string_dtype(train[col]):
            if isinstance(train.iloc[0, :][col], list):
                continue
            print(f'Обрабатываем {col}')
            train[col] = train[col].apply(text_processing.clean_bad_letters)

    train['title_parsed'] =  train['title_parsed'].str.replace('— РБК', '').str.strip()
    train['publish_date_mismatch_days'] = train['publish_date_parsed'] - train['publish_date']
    train['publish_date_mismatch_days'] = (
        train['publish_date_mismatch_days'].dt.total_seconds() / (24*3600)
        ).round(0).astype(int)

    train = train.join(get_url_and_sessions_count(train), how='left')
    train = extend_by_trading_features(train)
    train = extend_by_inflation_and_key_rate(train)

    train = extend_by_covid_features(train)

    train = extend_by_wheather(train)

    train['article_len'] = train['article_text_parsed'].apply(len)
    train['title_len'] = train['title'].apply(len)
    train['url_depth'] = train['url_parsed'].str.split('/').apply(len)
    train['authors_counts'] = train['authors'].apply(len)
    train['tags_counts'] = train['tags'].apply(len)

    return train.drop(columns=[
        'copyright_parsed',
        'datetime_parsed',
        'theme',
        'article_header_yandex_parsed']).sort_values(by=get_dt_features_from)


def get_neighbors_data(train: pd.DataFrame,
                       test: pd.DataFrame,
                       embeddings: pd.Series or pd.DataFrame,
                       keyed_vectors,
                       target_col: str,
                       offset: int=-1,
                       topn: int=5,
                       dt_col: str='publish_date') -> pd.DataFrame:
    """Для строк из теста находит в трейне topn наиболее похожих среди тех,
    что возникли в интервале offset от момента появления целевой строки (публикации новости)
    Для них считает среднюю косинусную близость к целевой строке по эмбеддингу и среднее число для целевой переменной
    Добавляет их как фичи в датасет

    Args:
        train (_type_): _description_
        test (_type_): _description_
        embeddings (_type_): _description_
        target_col (_type_): _description_
        offset (int, optional): _description_. Defaults to -1.
        topn (int, optional): _description_. Defaults to 5.
        dt_col (str, optional): _description_. Defaults to 'publish_date'.

    Returns:
        pd.DataFrame: _description_
    """
    col1 = f'top{topn}_neighbors_{target_col}'
    col2 = f'top{topn}_neighbors_cosine'

    neighbors_df = pd.DataFrame(
        index=test.index,
        columns=[f'top{topn}_neighbors_{target_col}', f'top{topn}_neighbors_cosine'],
        data = np.nan
        )

    for ind in tqdm(test.index):
        article_row = test.loc[ind]
        end = article_row[dt_col]

        start = end + pd.to_timedelta(f'{offset}D')

        if end < start:
            start, end = end, start

        slice_cond = (train[dt_col] <= end) & (train[dt_col] > start)
        slice_ind = train[slice_cond].index

        embeddings_in_slice = embeddings.loc[slice_ind]
        article_emb = embeddings.loc[ind].values
        similarities = \
            keyed_vectors.cosine_similarities(article_emb, embeddings_in_slice)
        top_n_sims = np.argsort(-similarities)[:topn]
        neighbors_df.loc[ind, col1] = train.loc[slice_ind[top_n_sims], target_col].mean()
        neighbors_df.loc[ind, col2] = np.mean(similarities)
    return neighbors_df


def get_high_correlated_features(data, threshold=0.95, method='spearman'):
    """Возвращает список скоррелированных выше порога признаков

    Args:
        data (_type_): _description_
        threshold (float, optional): _description_. Defaults to 0.95.
        method (str, optional): _description_. Defaults to 'spearman'.

    Returns:
        _type_: _description_
    """

    num_cols = sorted(data.select_dtypes(include=np.number).columns.tolist())

    num_cols = [col for col in num_cols if sorted(data[col].unique()) != [0,1]]

    data_for_corr = data[num_cols].copy()

    corr_matrix = data_for_corr.corr(method=method).abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    cols_to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    return cols_to_drop


def prepare_for_radar(train: pd.DataFrame,
                      train_cols: list[str],
                      drop_duplicates: bool=True,
                      duplicates_subset: Optional[list[str]]=None) -> tuple[list[str], pd.DataFrame]:
    """Готовит данные (cat_features и X) для ml_scripts.NewsRadar

    Args:
        train (pd.DataFrame): _description_
        train_cols (list[str]): _description_
        drop_duplicates (bool, optional): _description_. Defaults to True.
        duplicates_subset (Optional[list[str]], optional): _description_. Defaults to None.

    Returns:
        tuple[list[str], pd.DataFrame]: _description_
    """
    if duplicates_subset is None:
        duplicates_subset = ['url_id', 'ctr']
    if drop_duplicates:
        train = train.sort_values(
            by='days_from_collecting', ascending=False
            ).drop_duplicates(subset=['url_id', 'ctr'])

    # готовим категориальные признаки для бустингов
    cat_features = []
    for col in train_cols:
        try:
            train.loc[:, col] = pd.to_numeric(train[col])
        except ValueError:
            train.loc[:, col] = train[col].astype("category")
            cat_features.append(col)

    return cat_features, train
