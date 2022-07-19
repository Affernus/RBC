"""Парсер сайта РБК
"""

import os
from requests_html import HTMLSession
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import pyppeteer
from rbk.utils import helpers, prepare_data
from rbk.utils.global_vars import PARSED_DATA_PATH, PARSED_FILENAME, RAW_DATA_PATH, TRAIN_FILENAME, TEST_FILENAME

def get_soup(full_url: str, session: HTMLSession) -> BeautifulSoup:
    """Получает суп

    Args:
        base_url (str): _description_
        doc_id (str): _description_
        session (HTMLSession): _description_

    Returns:
        BeautifulSoup: _description_
    """

    try:
        page = session.get(full_url)
    except (TimeoutError, requests.exceptions.ConnectionError):
        print('session.get(full_url) TimeOutError or ConnectionError')
        return None

    try:
        page.raise_for_status()
    except ValueError:
        print(f'Dead link {page.status_code}')
        return None

    try:
        page.html.render()
    except pyppeteer.errors.TimeoutError:
        print(f'page.html.render {full_url} TimeOutError')

    page_raw = page.html.raw_html
    soup = BeautifulSoup(page_raw, 'html.parser')
    return soup


def parce_page_soup(soup: BeautifulSoup) -> list:
    """Парсит суп

    Args:
        soup (BeautifulSoup): _description_

    Returns:
        list: _description_
    """
    views = soup.find(class_="article__header__counter js-insert-views-count")
    if views:
        views = views.get_text()

    data_category = soup.find(class_="js-rbcslider-slide rbcslider__slide current")
    if data_category:
        data_category = data_category['data-category']

    datetime = soup.find(class_="article__header__date")
    if datetime:
        datetime = datetime['datetime']

    title = soup.find("meta", attrs={'name':'title'})
    if title:
        title = title['content']

    description = soup.find("meta", attrs={'name':'description'})
    if description:
        description = description['content']

    keywords = soup.find("meta", attrs={'name':'keywords'})
    if keywords:
        keywords = keywords['content']

    copy_right = soup.find("meta", attrs={'name':'copyright'})
    if copy_right:
        copy_right = copy_right['content']

    url = soup.find("meta", attrs={'property':'og:url'})
    if url:
        url = url['content']

    yandex_recommendations_category = soup.find(
        "meta", attrs={'property': 'yandex_recommendations_category'}
        )
    if yandex_recommendations_category:
        yandex_recommendations_category = yandex_recommendations_category['content']

    article_header_yandex = soup.find(class_="article__header__yandex")
    if article_header_yandex:
        article_header_yandex = article_header_yandex.get_text()

    all_tags = soup.find_all(class_="article__tags__item")
    tags = []
    if all_tags:
        tags = [t.get_text() for t in all_tags]

    article_text_free = soup.find(class_="article__text article__text_free")
    article_text = ''
    if article_text_free:
        article_text_free_p = article_text_free.find_all('p')
    if article_text_free_p:
        article_text = ' '.join(
            [el.get_text().strip() for el in article_text_free_p]
            ).strip().replace('\xa0', ' ')

    all_authors = soup.find_all(class_="article__authors__author__name")
    authors = []
    if all_authors:
        authors = [ath.get_text().strip(', ') for ath in all_authors]

    return [views, data_category, datetime, title,
            description, keywords, copy_right, url,
            yandex_recommendations_category,
            article_header_yandex, tags,
            article_text, authors]


def parse_all_ids(urls_data: pd.Series,
                   base_url: str='https://www.rbc.ru/rbcfreenews/',
                   path_to_save_pickle: str='parsing_result.pickle') -> dict:
    """Парсит айдишники новостей из датасета

    Args:
        urls_data (pd.Series): _description_
        base_url (_type_, optional): _description_. Defaults to 'https://www.rbc.ru/rbcfreenews/'.
        path_to_save_pickle (str, optional): _description_. Defaults to 'parsing_result.pickle'.

    Returns:
        dict: _description_
    """
    session = HTMLSession()
    res_dict = {}
    counter = 0
    urls_data = urls_data.copy()
    urls_data = base_url + urls_data

    for (ind, url) in tqdm(zip(urls_data.index, urls_data.values), total=urls_data.shape[0]):
        soup = get_soup(url, session)
        if soup:
            counter += 1
            res_dict[ind] = parce_page_soup(soup)
            if counter % 100 == 0:
                helpers.dump_pickle(res_dict, path_to_save_pickle)

    helpers.dump_pickle(res_dict, path_to_save_pickle)
    return res_dict


if __name__ == "__main__":
    # парсинг доп инфы о статьях на базе извлеченного из id линка на статью
    df_train = pd.read_csv(os.path.join(RAW_DATA_PATH, TRAIN_FILENAME), index_col=0)
    df_test = pd.read_csv(os.path.join(RAW_DATA_PATH, TEST_FILENAME), index_col= 0)
    df_all = pd.concat([df_train, df_test])
    all_url_ids = prepare_data.get_url_and_sessions_count(df_all)['url_id']

    parse_all_ids(all_url_ids,
                  base_url='https://www.rbc.ru/rbcfreenews/',
                  path_to_save_pickle=\
                    os.path.join(PARSED_DATA_PATH, PARSED_FILENAME))
