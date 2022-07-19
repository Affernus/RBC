"""Скрипты, относящиеся к построению и валидации моделей машинного обучения
"""
from typing import Optional
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np


from sklearn.base import clone, RegressorMixin, BaseEstimator
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from rbk.utils import prepare_data
from rbk.utils.global_vars import DROP_FROM_TRAIN


class MyStackRegressor(BaseEstimator, RegressorMixin):
    """Делает стек из estimators
    Если указан base_estimator, то строит прогноз, используя прогнозы estimators как фичи
    Если не указан, то прогноз строит как среднее по прогнозам estimators

    Args:
        BaseEstimator ([type]): [description]
        ClassifierMixin ([type]): [description]
    """


    def __init__(self,
                estimators_list: list,
                base_estimator=None,
                kfold_splits: int=10,
                random_state=None):

        self.estimators_list = estimators_list
        self.base_estimator = base_estimator
        self.random_state = random_state
        self.kfold_splits = kfold_splits
        self.feature_name_ = None


    def fit(self, X: pd.DataFrame, y: pd.Series):
        """обучение

        Args:
            X (pd.DataFrame): _description_
            y (pd.Series): _description_

        Returns:
            _type_: _description_
        """
        self.estimators_list = [clone(est) for est in self.estimators_list]
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        self.feature_name_ = X.columns

        # если base_estimator не None, то готовим для него обучающий датасет и обучаем его;
        # чтобы избежать утечки данных и адекватно оценить простые модели
        # датасет делаем из предиктов на kfoldss

        if self.base_estimator is not None:
            skf = StratifiedKFold(n_splits=self.kfold_splits,
                                  shuffle=True,
                                  random_state=self.random_state)
            splits = skf.split(X, y)
            base_estimator_train = pd.DataFrame(index=X.index)

            for train_index, valid_index in splits:
                x_train, y_train = X.loc[train_index, :], y.loc[train_index]
                x_val = X.loc[valid_index, :]

                for i, est in enumerate(self.estimators_list):
                    est.fit(x_train, y_train)
                    base_estimator_train.loc[valid_index, i] = est.predict(x_val)

            base_estimator_train['target'] = y

            self.base_estimator.fit(
                base_estimator_train.drop(columns='target'),
                base_estimator_train['target']
                )

        # обучаем модели на полном датасете
        for i, _ in enumerate(self.estimators_list):
            self.estimators_list[i].fit(X, y)

        return self


    def predict(self, X: pd.DataFrame) ->np.ndarray:
        """прогнозирование

        Args:
            X (pd.DataFrame): _description_

        Returns:
            np.ndarray: _description_
        """
        base_estimator_data = pd.DataFrame(index=X.index)

        for i, _ in enumerate(self.estimators_list):
            base_estimator_data[i] = \
                self.estimators_list[i].predict(X)

        if self.base_estimator is None:
            predictions = base_estimator_data.mean(axis=1).values
            return predictions

        return self.base_estimator.predict(base_estimator_data)


    def fit_predict(self, X, y, **fit_params):
        return self.fit(X, y, **fit_params).predict(X)


class NewsRadar(BaseEstimator, RegressorMixin):
    """Класс объединяет обучение и применение
    ML-модели и эвристической модели

    Args:
        BaseEstimator (_type_): _description_
        RegressorMixin (_type_): _description_
    """
    def __init__(self, estimator,
                heuristic_features: Optional[list[str]]=None,
                estimator_features: Optional[list[str]]=None,
                round_numeric_to=2) -> None:
    
        self.estimator = estimator
        self.heuristic_data = pd.DataFrame()
        self.heuristic_features = heuristic_features
        self.estimator_features = estimator_features
        self.round_numeric_to = round_numeric_to
        self.outliers_threshold = None

    @staticmethod
    def prepare_to_heuristic(X: pd.DataFrame,
                            heuristic_features: list[str],
                            round_numeric_to: int=2) -> pd.DataFrame:
        """_summary_

        Args:
            X (pd.DataFrame): _description_
            y (pd.Series): _description_
            heuristic_features (list[str]): _description_

        Raises:
            TypeError: _description_
            TypeError: _description_
            ValueError: _description_

        Returns:
            pd.DataFrame: _description_
        """
        heur = X[heuristic_features].copy()

        if not isinstance(X, pd.DataFrame):
            raise TypeError(f'X должен быть pandas DataFrame, получено {type(X)}')

        for ftr in heuristic_features:
            if is_numeric_dtype(heur[ftr]):
                heur[ftr] = heur[ftr].round(round_numeric_to).astype(str)

        return heur


    @staticmethod
    def get_heuristic(X: pd.DataFrame, y: pd.Series,
                      heuristic_features: list[str],
                      round_numeric_to: int=2) -> pd.DataFrame:
        """_summary_

        Args:
            X (pd.DataFrame): _description_
            y (pd.Series): _description_
            heuristic_features (list[str]): _description_

        Raises:
            TypeError: _description_
            TypeError: _description_
            ValueError: _description_

        Returns:
            pd.DataFrame: _description_
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f'X должен быть pandas DataFrame, получено {type(X)}')

        if not isinstance(y, pd.Series):
            raise TypeError(f'y должен быть pandas Series, получено {type(y)}')

        if (X.index != y.index).sum() > 0:
            raise ValueError('Индексы X и y не совпадают')

        if heuristic_features is None:
            heuristic_features = ['url_id', 'ctr']

        heur = NewsRadar.prepare_to_heuristic(
            X, heuristic_features, round_numeric_to=round_numeric_to
            )

        if (heur.index != y.index).sum() > 0:
            raise ValueError('Индексы после подготовки эвристик не совпадают')

        heur['target'] = y
        heuristic_data = heur.groupby(heuristic_features).mean().reset_index()

        return heuristic_data

    def fit(self, X: pd.DataFrame, y: pd.Series, outliers_threshold: float=None):
        """Обучение estimator и получение эвристик

        Args:
            X (pd.DataFrame): _description_
            y (pd.Series): _description_
            heuristic_features (Optional[list[str]], optional): _description_. Defaults to None.
            estimator_features (Optional[list[str]], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
            KeyError: _description_

        Returns:
            _type_: _description_
        """
        self.estimator = clone(self.estimator)
        self.outliers_threshold = outliers_threshold

        if not isinstance(X, pd.DataFrame):
            raise TypeError(f'X должен быть pandas DataFrame, получено {type(X)}')

        if not isinstance(y, pd.Series):
            raise TypeError(f'y должен быть pandas Series, получено {type(y)}')

        if (X.index != y.index).sum() > 0:
            raise ValueError('Индексы X и y не совпадают')

        if self.heuristic_features is None:
            self.heuristic_features = ['url_id', 'ctr']

        if self.estimator_features is None:
            self.estimator_features = X.columns

        nf_cols_set = set(self.heuristic_features).difference(set(X.columns))
        if nf_cols_set != set():
            raise KeyError(f'В X нет колонок {nf_cols_set}, заявленных для расчета эвристик')

        heur = X[self.heuristic_features].copy()
        heur['target'] = y
        self.heuristic_data = self.get_heuristic(X, y,
                                                self.heuristic_features,
                                                round_numeric_to=self.round_numeric_to)
    
        if self.outliers_threshold is not None:
            X = X.loc[y<=self.outliers_threshold, :].copy()
            y = y[y<=self.outliers_threshold]

        self.estimator.fit(X[self.estimator_features], y)

        return self


    def predict(self, X: pd.DataFrame, use_heuristic=True) -> np.ndarray:
        """Совместный прогноз по эвристикам и по ml-модели

        Args:
            X (pd.DataFrame): _description_
        """
        nf_cols_set = set(self.heuristic_features).difference(set(X.columns))
        if nf_cols_set != set():
            raise KeyError(f'В X нет колонок {nf_cols_set}, заявленных для расчета эвристик')

        if use_heuristic:
            heur = self.prepare_to_heuristic(X, self.heuristic_features,
                                            round_numeric_to=self.round_numeric_to)
            index_name = heur.index.name
            if index_name is None:
                index_name = 'index'
            heur = heur.reset_index()
            heuristic_predicted = heur.merge(
                self.heuristic_data, on=self.heuristic_features, how='inner'
                )
            heuristic_predicted = heuristic_predicted.set_index(index_name)['target']
        else:
            heuristic_predicted = pd.Series(dtype=float)

        heuristic_predicted.name = 'target'

        heuristic_index = heuristic_predicted.index
        ml_index = X.index[~X.index.isin(heuristic_index)]
        ml_data = X.loc[ml_index, self.estimator_features].copy()

        if ml_data.shape[0] != 0:
            ml_predicted = self.estimator.predict(ml_data)
            ml_predicted = pd.Series(index=ml_index, data=ml_predicted)
        else:
            ml_predicted = pd.Series(dtype=float)

        ml_predicted.name = 'target'
        result = pd.concat([heuristic_predicted, ml_predicted])
        return result[X.index].values


    def fit_predict(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> np.ndarray:
        """_summary_

        Args:
            X (pd.DataFrame): _description_
            y (pd.Series): _description_

        Returns:
            np.ndarray: _description_
        """
        return self.fit(X, y, **fit_params).predict(X)


def select_best_features(df_train: pd.DataFrame,
                        target_col: str,
                        lgbm_regressor_pars: Optional[dict]=None,
                        k_best=100,
                        n_splits: int=50,
                        test_size=.3,
                        cols_to_drop: Optional[list[str]]=None,
                        outliers_threshold: Optional[float]=None) -> tuple[list[str], pd.DataFrame]:
    """Отбирает фичи для модели на кросс-валидациях
    В качестве базовой модели используется облегченный LGBMRegressor
    с небольшим числом решателей (100-200) и шагом порядка 0.1

    Args:
        train (pd.DataFrame): _description_
        lgbm_regressor (LGBMRegressor): _description_
        n_splits (int, optional): _description_. Defaults to 50.
        cols_to_drop (Optional[list[str]], optional): _description_. Defaults to None.

    Returns:
        list[str]: _description_
    """
    if lgbm_regressor_pars is None:
        lgbm_regressor_pars = {}

    lgbm_regressor = LGBMRegressor(**lgbm_regressor_pars)

    if cols_to_drop is None:
        cols_to_drop = DROP_FROM_TRAIN

    train_cols = [col for col in df_train.columns if col not in DROP_FROM_TRAIN]
    all_importances = pd.DataFrame(index=train_cols)
    print('Feature selection started...')
    for counter in range(n_splits):
        print(f'{(counter+1):<5} из {n_splits}', end='\r')
        train, _ = train_test_split(df_train, test_size=test_size, random_state=counter)

        _, train = prepare_data.prepare_for_radar(train,
                                                  train_cols,
                                                  drop_duplicates=True,
                                                  duplicates_subset=None)
        if outliers_threshold is not None:
            train = train[train[target_col] <= outliers_threshold]

        stage_regressor = clone(lgbm_regressor)
        stage_regressor.fit(train[train_cols], train[target_col])

        # # считаем importances
        importances_df = pd.DataFrame(
            data=stage_regressor.feature_importances_,
            index=train_cols,
            columns=[f'importance{counter}'])

        all_importances = all_importances.join(importances_df, how='outer')

    all_importances['imortance_mean'] = all_importances.mean(axis=1)
    all_importances['imortance_std'] = all_importances.std(axis=1)
    all_importances = all_importances.sort_values(by='imortance_mean', ascending=False)

    best_features = all_importances.index[:k_best].to_list()
    print('\nDone!')
    return best_features, all_importances


def my_cross_validation(df_train: pd.DataFrame,
                        train_cols: list[str],
                        target_col: str,
                        pars_lgbm: dict,
                        pars_cb: dict,
                        outliers_threshold: float=None,
                        n_splits: int=50) -> tuple[list, pd.DataFrame, pd.DataFrame]:
    """Кросс-валидация модели на базе стека из lgmb и catboost
    возвращает для каждого шага валидации вклад строки в r2,
    средние и std метрик - r2 и rmse
    средние и std feature_importance от lightgbm

    Args:
        df_train (pd.DataFrame): _description_
        train_cols (list[str]): _description_
        target_col (str): _description_
        pars_lgbm (dict): _description_
        pars_cb (dict): _description_
        outliers_threshold (float, optional): _description_. Defaults to None.
        n_splits (int, optional): _description_. Defaults to 50.

    Returns:
        tuple[list, pd.DataFrame, pd.DataFrame]: _description_
    """
    cv_score_val = []
    cv_score_train = []
    cv_rmse_val = []
    cv_rmse_train = []

    all_errors = []

    all_importances = pd.DataFrame(index=train_cols)

    for counter in range(n_splits):
        print(f'{counter:<5}', end='\r')

        # разбиваем выборку и готовим трейн и тест
        train, val = train_test_split(df_train, test_size=.3, stratify=None, random_state=counter+1)

        cat_features, train = prepare_data.prepare_for_radar(train,
                                                            train_cols,
                                                            drop_duplicates=True,
                                                            duplicates_subset=None)

        _, val = prepare_data.prepare_for_radar(val,
                                                train_cols,
                                                drop_duplicates=False,
                                                duplicates_subset=None)
        pars_cb['cat_features'] = cat_features

        clf1 = LGBMRegressor(**pars_lgbm)
        clf2 = CatBoostRegressor(**pars_cb)

        my_model = MyStackRegressor([clf1, clf2])
        my_radar = NewsRadar(my_model,
                            heuristic_features=['url_id', 'ctr'],
                            estimator_features=train_cols,
                            round_numeric_to=3)

        my_radar.fit(train, train[target_col], outliers_threshold=outliers_threshold)

        val_predicted = pd.Series(data=my_radar.predict(val), index=val.index)
        train_predicted = \
            pd.Series(data=my_radar.predict(train, use_heuristic=False), index=train.index)

        # считаем importances
        importances_df = pd.DataFrame(
            data=my_radar.estimator.estimators_list[0].feature_importances_,
            index=train_cols,
            columns=[f'importance{counter}'])

        all_importances = all_importances.join(importances_df, how='outer')

        # считаем метрики и некоторые статистики
        # ---------------------------------------------------------------------------------------------
        y_val = val[target_col]
        y_train = train[target_col]
        cv_score_val.append(r2_score(y_val, val_predicted))
        cv_score_train.append(r2_score(y_train, train_predicted))
        cv_rmse_val.append(mean_squared_error(y_val, val_predicted)**.5)
        cv_rmse_train.append(mean_squared_error(y_train, train_predicted)**.5)    

        predicted_df = df_train.loc[val.index, :]
        predicted_df['predicted_target'] = val_predicted
        predicted_df['predicted_err'] = predicted_df[target_col] - predicted_df['predicted_target']

        mean_target = predicted_df[target_col].mean()

        predicted_df['diff_mean'] = predicted_df[target_col] - mean_target

        predicted_df['r2'] = predicted_df['predicted_err']**2 / (predicted_df['diff_mean']**2).sum()
        predicted_df['r2'] = predicted_df['r2']

        predicted_df = predicted_df.set_index(target_col)
        all_errors.append(predicted_df['r2'])
        # ---------------------------------------------------------------------------------------------

    cv_results_df = pd.DataFrame([cv_score_val, cv_score_train, cv_rmse_val, cv_rmse_train]).T
    cv_results_df.columns = ['cv_score_val', 'cv_score_train', 'cv_rmse_val', 'cv_rmse_train']

    mean_res = cv_results_df.mean().reset_index().set_index('index')
    mean_res.columns = ['mean']

    std_res = cv_results_df.std().reset_index().set_index('index')
    std_res.columns = ['std']

    agg_res = mean_res.join(std_res).round(4)

    all_importances['imortance_mean'] = all_importances.mean(axis=1)
    all_importances['imortance_std'] = all_importances.std(axis=1)

    return all_errors, agg_res, all_importances[['imortance_mean', 'imortance_std']],\
        y_val, val_predicted, y_train, train_predicted
