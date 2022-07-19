"""Скрипты для обработки текстов
"""
import re
import html
import unicodedata

from collections import Counter
from typing import Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
from pymystem3 import Mystem
from rbk.utils.global_vars import RUS_LETTERS, BAD_LETTERS_MAP, RUS_VOWELS, ENG_VOWELS


def clean_bad_letters(text):
    """В русских текстах встречаются символы '̆ и '̈, убираем их

    Args:
        text_series (_type_): _description_

    Returns:
        _type_: _description_
    """
    if not isinstance(text, str):
        return text
    cleaned_text = text.strip()
    for bad_letter, repl in BAD_LETTERS_MAP.items():
        cleaned_text = cleaned_text.replace(bad_letter, repl)
    return cleaned_text


def clean_all_symbols(text: str, exceptions: Optional[str]='') -> str:
    """_summary_

    Args:
        text (str): _description_

    Returns:
        str: _description_
    """

    prepared = text.lower()
    prepared = clean_bad_letters(prepared)
    prepared = prepared.replace('\\n', '')
    cleaned = re.sub(f'[^a-z{RUS_LETTERS}{exceptions}]', ' ', prepared)
    return ' '.join(cleaned.split())


def normalize_text(raw, encode='ASCII', errors='ignore', form='NFKD'):
    """
    function for ascii conversion
    """
    nfkd_form = unicodedata.normalize(form, raw)
    result = nfkd_form.encode(encode, errors).decode(encode)
    return result


def lemmatize_text(text: str, lemmatizer: Mystem, stop_words: Optional[list[str]]=None) -> str:
    """_summary_

    Args:
        text (str): _description_
        lemmatizer (Mystem): _description_
        stop_words (Optional[list[str]], optional): _description_. Defaults to None.

    Returns:
        str: _description_
    """
    lemmas = lemmatizer.lemmatize(text)
    if stop_words is not None:
        lemmas = [lem for lem in lemmas if lem.lower() not in stop_words]
    return ''.join(lemmas)


def get_tf(text: str or list[str]) -> dict:
    """_summary_

    Args:
        text (str): _description_

    Returns:
        dict: _description_
    """
    if isinstance(text, str):
        splited_text = text.split()
    elif isinstance(text, list):
        splited_text = text
    else:
        raise ValueError(f'unknown text format {type(text)}')

    all_words_count = len(splited_text)
    word_freq_dict = Counter(splited_text)
    for word in word_freq_dict:
        word_freq_dict[word] /= all_words_count
    return word_freq_dict


def get_idf(corpus: pd.Series) -> tuple[dict, dict]:
    """Считает idf слов для корпуса текстов

    Args:
        corpus (pd.Series): _description_

    Raises:
        ValueError: _description_

    Returns:
        dict: _description_
    """
    doc_freq = {}
    corpus_size = corpus.shape[0]
    for text in corpus:
        if isinstance(text, str):
            words_in_text = text.split()
        elif isinstance(text, list):
            words_in_text = text
        else:
            raise ValueError(f'unknown text format {type(text)}')
        words_in_text = list( set(words_in_text) )
        for word in words_in_text:
            if word in doc_freq:
                doc_freq[word] += 1 / corpus_size
            else:
                doc_freq[word] = 1 / corpus_size
    idf = {}
    for word, freq in doc_freq.items():
        idf[word] = np.log( 1 / freq )

    return idf, doc_freq


def get_tf_idf_for_text(text: str or list[str], idf: dict) -> dict:
    """Считает tf-idf слов для конкретного текста из корпуса

    Args:
        text (strorlist[str]): _description_
        idf (dict): _description_

    Raises:
        KeyError: _description_

    Returns:
        dict: _description_
    """
    tf = get_tf(text)
    tf_idf = {}

    for word, freq in tf.items():
        if word not in idf:
            print(f'{word} отсутствует в idf')
            w_idf = max(idf.values())
        else:
            w_idf = idf[word]
        tf_idf[word] = freq * w_idf

    return tf_idf


def get_tf_idf_for_clusters(corpus: pd.Series,
                            texts_indexes: Optional[list or pd.Series]=None,
                            lemmatizer: Mystem=None,
                            stop_words: Optional[list]=None,
                            use_idf: bool=True) -> dict:
    """_summary_

    Args:
        corpus (pd.Series): _description_
        texts_indexes (Optional[list or pd.Series], optional): _description_. Defaults to None.
        lemmatizer (Mystem, optional): _description_. Defaults to None.
        stop_words (Optional[list], optional): _description_. Defaults to None.
        use_idf (bool, optional): _description_. Defaults to True.

    Raises:
        TypeError: _description_

    Returns:
        dict: _description_
    """

    tqdm.pandas()
    if not isinstance(corpus, pd.Series):
        raise TypeError(f'corpus must be pd.Series, {type(corpus)} found')

    corpus = corpus.copy()

    print('text cleaning started')
    corpus = corpus.apply(clean_all_symbols)
    print('text cleaning done')

    if lemmatizer is not None:
        print('lemmatization started')
        corpus = corpus.progress_apply(lemmatize_text, args=(lemmatizer,), stop_words=stop_words)
        print('lemmatization done')

    print('tf-idf started')

    if texts_indexes is None:
        texts_indexes = list(range(0, len(corpus)))

    if isinstance(texts_indexes, pd.Series):
        texts_indexes = texts_indexes.to_list()

    texts_list = corpus.to_list()
    all_texts = ' '.join(texts_list)
    wf_global = get_tf(all_texts)
    tf_idf = {}
    for ind, text in zip(texts_indexes, texts_list):
        tf_idf[ind] = get_tf(text)
        if use_idf:
            for word in tf_idf[ind]:
                tf_idf[ind][word] *= np.log( 1 / wf_global[word] )
            tf_idf[ind] = pd.Series(tf_idf[ind], dtype=float).sort_values(ascending=False)
    print('all done!')

    return tf_idf


def prepare_corpus_for_readability(raw_corpus: pd.Series, lemmatizer: Mystem) -> pd.Series:
    """Готовит корпус для расчета фич читабельности

    Args:
        raw_corpus (pd.Series): _description_
        lemmatizer (Mystem): _description_

    Returns:
        pd.Series: _description_
    """
    corpus = raw_corpus.copy()
    corpus = corpus.apply(html.unescape)
    corpus = corpus.progress_apply(clean_all_symbols, exceptions='.')
    corpus = corpus.str.replace('.', ' . ', regex=False)
    corpus = corpus.str.split().apply(' '.join)
    corpus = corpus.progress_apply(lemmatize_text, lemmatizer=lemmatizer, stop_words=None)
    return corpus


def calc_asl(text: str) -> float:
    """average number of words per sentence (ASL)

    Args:
        text (str): _description_

    Returns:
        float: _description_
    """
    words = text.replace('.', ' ').split()
    sentences = text.split('.')
    n_words = len(words)
    n_sentences = len(sentences)
    return n_words / n_sentences


def calc_asw(text: str) -> float:
    """average number of syllables per word (ASW)

    Args:
        text (str): _description_

    Returns:
        float: _description_
    """
    words = text.replace('.', ' ').split()
    vowels = re.sub(f'[^{RUS_VOWELS}{ENG_VOWELS}]', '', text)
    n_vowels = len(vowels)
    n_words = len(words)
    return n_vowels / n_words


def calc_plw(text: str) -> float:
    """percentage of long words in text (PLW)

    Args:
        text (str): _description_

    Returns:
        float: _description_
    """
    words = text.replace('.', ' ').split()
    
    n_words = len(words)
    long_words = [w for w in words if len(re.sub(f'[^{RUS_VOWELS}{ENG_VOWELS}]', '', w)) >= 4]
    n_long_words = len(long_words)
    return n_long_words / n_words


def calc_ttr(text: str) -> float:
    """type-token ratio (TTR)

    Args:
        text (str): _description_

    Returns:
        float: _description_
    """
    words = text.replace('.', ' ').split()
    uniq_words = set(words)
    return len(uniq_words) / len(words)
