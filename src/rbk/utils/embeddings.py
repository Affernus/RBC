"""Модуль для расчета и обработки эмбеддингов и модели word2vec
"""
import os
import copy
import multiprocessing
from typing import Optional

import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from rbk.utils import text_processing
from rbk.utils.global_vars import ROOT_ABSPATH, MODELS_PATH


def calc_summary_embedding(sequence: list,
                           model: Optional[KeyedVectors]=None,
                           idf: Optional[dict]=None,
                           weights: Optional[list[float]]=None,
                           reject: Optional[int]=0) -> np.ndarray:
    """Считает эмбеддинг события на основе эмбеддингов доменов
    Эмбеддинги доменов берутся из модели, обученной на корпусе
    событий по всему набору данных
    или подаются as is"""

    if not isinstance(sequence, (list, np.ndarray)):
        return np.nan
    if len(sequence)==0:
        return np.nan

    if not isinstance(reject, int):
        raise ValueError('reject должен быть int')

    sequence = copy.deepcopy(sequence)

    if reject !=0:
        sequence = sequence[:reject]

    if len(sequence) == 0:
        return np.nan

    if model is not None:
        # если передана модель, получаем вектора слов
        # если не передана, то счтитаем, что в sequence
        # содержались именно вектора а не слова
        words_embs = [model[word] if word in model else np.zeros(model.vector_size) for word in sequence]
        if weights is None:
            if idf is None:
                weights = [1 for _ in words_embs]
            else:
                tf_idf = text_processing.get_tf_idf_for_text(sequence, idf)
                weights = [tf_idf[wrd] for wrd in sequence]
    else:
        # если переданы эмбеддинги, то idf не рассматриваем
        if weights is None:
            weights = [1 for _ in sequence]
        words_embs = sequence

    # взвешиваем эмбеддинги на веса
    # весами могут быть индексы цитируемости и проч.
    # нормировочный коэффициент веса
    weight_scale_coef = sum(weights)
    weights = [
        wgt/weight_scale_coef
        for wgt in weights
        ]
    words_embs = [emb * wgt for wgt, emb in zip(weights, words_embs)]

    return np.sum(words_embs, axis=0)


class GetTrainModel():
    def __init__(self,
                model_name='w2v_model',
                model_pars=None,
                retrain_model=False
                ):

        self.models_dir = os.path.join(ROOT_ABSPATH, MODELS_PATH, 'word2vec')
        self.model_file = os.path.join(self.models_dir, f'{model_name}.model')

        self.model_name = model_name
        self.model_pars = model_pars
        self.retrain_model = retrain_model
        self.model = None


    def get_train_model(self, corpus=None):
        """Загружает уже обученную модель word2vec или обучает её заново

        Args:
            corpus (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if self.model_pars is None:
            self.model_pars = \
                dict(min_count=5,
                     window=5,
                     vector_size=100,
                     negative=5,
                     epochs=5)

        if not os.path.isfile(self.model_file) or self.retrain_model:
            if corpus is None:
                raise ValueError('Для обучения модели нужен корпус')
            print(f'В {self.models_dir} нет файла модели {self.model_name}.model или retrain_model=True, модель будет обучена заново')
            cores = multiprocessing.cpu_count()
            model = Word2Vec(sentences=corpus,
                                workers=cores-1,
                                **self.model_pars)
            model.save(self.model_file)
        else:
            model = Word2Vec.load(self.model_file)
        self.model = model
        return model


def encode_category_by_embs(embs, data, cat_col):
    """Кодирует заданные категории средним эмбеддингом

    Args:
        embs (_type_): _description_
        data (_type_): _description_
        cat_col (_type_): _description_

    Returns:
        _type_: _description_
    """
    exploded_category = data[[cat_col]].explode(column=cat_col)
    embs.name = 'emb'
    result = exploded_category.join(embs)
    result = result.groupby(cat_col)['emb'].mean()
    return result


def get_embs_df(embs, prefix=''):
    """Получает из сохраненных эмбеддингов датафреймы

    Args:
        embs (_type_): _description_
        prefix (str, optional): _description_. Defaults to ''.

    Returns:
        _type_: _description_
    """
    embs_df = embs.apply(pd.Series)
    embs_df.columns = [f'{prefix}{col}' for col in embs_df.columns]
    return embs_df
