"""Библиотека содержит глобальные переменные и константы
"""
import os

RANDOM_STATE = 27

# В ОБЩЕМ СУЛЧАЕ МЕНЯТЬ НУЖНО ТОЛЬКО ROOT_ABSPATH!
# --------------------------------------------------------->>>>>
# путь к корневой директории проекта
ROOT_ABSPATH = '/Users/affernus/PROJECTS/hacks/RBC'
# <<<<<---------------------------------------------------------

# relative path относительно кореновой директории к сырым данным
RAW_DATA_PATH = os.path.join('data', 'raw')

# relative path относительно кореновой директории к подготовленным данным
PREPARED_DATA_PATH = os.path.join('data', 'prepared')

# relative path относительно кореновой директории к папке с эмбеддингами
EMBEDDINGS_PATH = os.path.join('data', 'embeddings')

# relative path относительно кореновой директории к папкам внешних данных
PARSED_DATA_PATH = os.path.join('data', 'external', 'parsing')
ECONOMIC_DATA_PATH = os.path.join('data', 'external', 'economic')
COVID_PATH = os.path.join('data', 'external', 'covid')
WEATHER_PATH = os.path.join('data', 'external', 'weather')

# relative path относительно кореновой директории к папке с моделями
MODELS_PATH = os.path.join('src', 'models')

TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'
PARSED_FILENAME = 'parsing_train_test.pickle'

COVID_ISO_CODE = ['OWID_EUN', 'RUS', 'USA']
COVID_COLS = ['date', 'new_cases_per_million', 'new_deaths_per_million']

RUS_VOWELS = 'аяуюоеёэиы'
ENG_VOWELS = 'aeiou'

BAD_I = 'и'+'̆'
BAD_YO = 'е'+'̈'

BAD_LETTERS_MAP = {
    BAD_I: 'й',
    BAD_YO: 'ё',
    BAD_I.upper(): 'Й',
    BAD_YO.upper(): 'Ё'}

ECONOMIC_TRADING_NAMES = [
    'brent',
    'dax',
    'ftse',
    'gold_gcq2',
    'irts',
    'nasdaq',
    'nikkei',
    'sp500',
    'usd_rub'
    ]

ECONOMIC_COLS_MAPPING = {
    'Дата': 'date',
    'Цена': 'price',
    'Откр.': 'price_open',
    'Макс.': 'max_price',
    'Мин.': 'min_price',
    'Объём': 'volume',
    'Изм. %': 'growth_perc',
    'Ключевая ставка, % годовых': 'rf_key_rate_perc',
    'Инфляция, % г/г': 'rf_inflation_perc',
    'Цель по инфляции': 'rf_inflation_target'
}

ECONOMIC_NUM_COLS = [
    'Цена', 'Откр.', 'Макс.', 'Мин.', 'Изм. %'
    ]

MAX_SENTENCE_LEN = 10**3

PARSING_COLS = [
    'views',
    'data_category',
    'datetime',
    'title',
    'description',
    'keywords',
    'copyright',
    'url',
    'yandex_recommendations_category',
    'article_header_yandex',
    'tags',
    'article_text',
    'authors']

RUS_LETTERS_UPPER = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
RUS_LETTERS_LOWER = RUS_LETTERS_UPPER.lower()
RUS_LETTERS = RUS_LETTERS_UPPER + RUS_LETTERS_LOWER

DROP_FROM_TRAIN = [
    'publish_date',
    'session',
    'authors',
    'tags',
    'title',
    'local_hour',
    'views_parsed', # важно - мы не будет использовать просмотры, спарсенные с сайта rbc.ru, ни в обучении, ни в валидации!
    'data_category_parsed',
    'title_parsed',
    'description_parsed',
    'keywords_parsed',
    'url_parsed',
    'tags_parsed',
    'article_text_parsed',
    'authors_parsed',
    'publish_date_parsed',
    'days_from_parsing',
    'publish_date_mismatch_days',
    'url_id'
    ] + ['views', 'depth', 'full_reads_percent']
