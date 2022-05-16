from typing import List

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
from .feature_importance import FeatureImportance

__all__ = ('ForwardPermutationFeatureSelection',)


class ForwardPermutationFeatureSelection:
    """
    PermutationImportance: на тренировочных данных обучается модель и вычисляется значение метрики
        на контрольных данных. После этого к каждому признаку применялась случайная перестановка n-раз и
        с помощью обученной модели осуществляется предсказание и сравнение полученной метрики с исходным
        значением метрики без перемешивания.

        Признаки сортируются по убыванию значения важности, полученными с помощью метода PermutationImportance.
        Затем признаки добавлялись последовательно. При отсутствии улучшений модели на фиксированном количестве
        шагов происходит останов алгоритма.
    """
    def __init__(self, n: int = 5, epsilon: float = 0.1, permutation_importance_df: pd.DataFrame = None,
                 metric: str = 'roc_auc', verbose: bool = True, early_stopping_rounds: int = 10):
        """
        n:
            Количество перемешиваний колонки
        epsilon:
            Прирост метрики, который считается незначительным
        permutation_importance_df:
            Посчитанные значения PermutationImportance для каждого призака
            (неообязательнй параметр)
        metric:
            Метрика модели
        verbose:
            Вывод лога обучения модели
        early_stopping_rounds:
            Количество фичей, в течение которых наблюдается незначительный пророст метрики
        """
        self.epsilon = epsilon
        self.n = n
        self.permutation_importance_df = permutation_importance_df
        self.early_stopping_rounds = early_stopping_rounds
        self.metric_name = metric
        self.metric = get_scorer(metric)
        self.verbose = verbose
        self.subsets_ = []

    def fit_transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                      model: BaseEstimator) -> List:
        """
        Parameters
        ----------
        X_train:
            Данные для обучения
        X_test:
            Тестовый набор
        y_train:
            Целевая для обучающего набора
        y_test:
            Целевая для тестового набора
        model:
            Модель, совместимая с sklearn estimator

        Return value
        ------------
        Выбранный набор признаков
        """
        if self.permutation_importance_df is None:
            self.__fe = FeatureImportance(X_train, X_test, y_train, y_test, model, self.metric_name)
            self.permutation_importance_df = self.__fe.get_n_permutation_importance(self.n)

        self.permutation_importance_df = self.permutation_importance_df.sort_values('permutation_' + self.metric_name,
                                                                                    ascending=False)

        selected_features = []
        for i, col in enumerate(self.permutation_importance_df['features']):
            selected_features.append(col)
            if self.verbose:
                print('Fitting model on {0} features'.format(i + 1))
            model.fit(X_train[selected_features], y_train)
            current_metric = self.metric(model, X_test[selected_features], y_test)
            if self.verbose:
                print(self.metric_name + ' = {0}'.format(current_metric))
            self.subsets_.append({
                'score_' + self.metric_name: current_metric,
                'feature_names': list(selected_features)
            })

            if i > self.early_stopping_rounds and current_metric - self.subsets_[i - self.early_stopping_rounds] \
                    ['score_' + self.metric_name] < self.epsilon:
                break

        return selected_features[:-self.early_stopping_rounds]
