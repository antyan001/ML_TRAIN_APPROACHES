from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, cross_val_score

from .feature_importance import FeatureImportance

__all__ = ('ForwardPermutationFeatureSelectionCV',)


class ForwardPermutationFeatureSelectionCV:
    """
    PermutationImportance: на тренировочных данных обучается модель и вычисляется значение метрики
    на контрольных данных. После этого к каждому признаку применялась случайная перестановка n-раз и
    с помощью обученной модели осуществляется предсказание и сравнение полученной метрики с исходным
    значением метрики без перемешивания.

    Признаки сортируются по убыванию значения важности, полученными с помощью метода PermutationImportance.
    Затем признаки добавлялись последовательно. При отсутствии улучшений модели на фиксированном количестве
    шагов происходит останов алгоритма.
    """
    def __init__(self, n: int = 5, epsilon: float = 0.1, metric: str = 'roc_auc', verbose: bool = True,
                 early_stopping_rounds: int = 10, cv: int = 3, n_jobs: int = 1):
        """
        Parameters
        ---------
        n:
            Количество перемешиваний колонки
        epsilon:
            Прирост метрики, который считается незначительным
        permutation_importance_df:
            Посчитанные значения PermutationImportance для каждого призака необязательнй параметр)
        metric:
            Метрика модели
        verbose:
            Вывод лога обучения модели
        early_stopping_rounds:
            Количество фичей, в течение которых наблюдается незначительный пророст метрики
        cv:
            Количество фолдов для кроссвалидации
        n_jobs:
            Количество потоков для кросс валидации

        Attributes
        ---------
        subsets_: массив из выбранных на i итерации призаков и полученная метрика
        """
        self.epsilon = epsilon
        self.n = n
        self.early_stopping_rounds = early_stopping_rounds
        self.metric_name = metric
        self.metric = get_scorer(metric)
        self.folds = KFold(n_splits=cv, random_state=42)
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.subsets_ = []

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model: BaseEstimator) -> List:
        """
        Parameters
        ---------
        X:
            Набор данных для модели
        y:
            Целевая переменная
        model:
            Модель, совместимая с sklearn estimator

        Return value
        ------------
        Выбранный набор признаков
        """

        self.permutation_importance_df = None
        for i, (train_index, test_index) in enumerate(self.folds.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            self.__fe = FeatureImportance(X_train, X_test, y_train, y_test, model, self.metric_name)
            if self.permutation_importance_df is None:
                self.permutation_importance_df = self.__fe.get_n_permutation_importance(self.n)
            else:
                self.permutation_importance_df = self.permutation_importance_df \
                    .merge(self.__fe.get_n_permutation_importance(self.n), on='features', suffixes=('', '_' + str(i)))
        self.permutation_importance_df['permutation_mean'] = self.permutation_importance_df.mean(axis=1)
        self.permutation_importance_df = self.permutation_importance_df.sort_values('permutation_mean', ascending=False)

        selected_features = []
        for i, col in enumerate(self.permutation_importance_df['features']):
            selected_features.append(col)
            if self.verbose:
                print('Fitting model on {0} features'.format(i + 1))
            scores = cross_val_score(model, X[selected_features], y, scoring=self.metric,
                                    cv=self.folds, n_jobs=self.n_jobs)

            current_metric = scores.mean()
            if self.verbose:
                print(self.metric_name + ' = {0}'.format(current_metric))
            self.subsets_.append({
                'score_' + self.metric_name: current_metric,
                'feature_names': list(selected_features)
            })

            if i > self.early_stopping_rounds and current_metric - self.subsets_[i - self.early_stopping_rounds]['score_'+ self.metric_name] < self.epsilon:
                return selected_features[:-self.early_stopping_rounds]
        return selected_features

    def plot_features_importance(self, filename: str = 'permutation_mean_cv', n_features: int = 50,
                                 seaborn_palette: str = 'GnBu_d') -> None:
        """
        Построение графика важности признаков.

        Parameters
        ---------
        filename:
            Имя файла ддя сохранения графика важности
        n_features:
            Количество признаков для вывода на графике
        seaborn_palette:
            Имя палитры seaborn
        """
        self.__fe.plot_features_importance(self.permutation_importance_df, 'permutation_mean', filename,
                                           n_features, seaborn_palette)
