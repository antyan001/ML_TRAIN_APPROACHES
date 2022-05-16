from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, precision_score, \
    recall_score, confusion_matrix, \
    get_scorer
from sklearn.preprocessing import binarize

__all__ = ('FeatureImportance')


class FeatureImportance:
    """
    В классе реализованы методы исследования важности фич:
    Leave_one_out - поочередное удаление одного из признаков
    One_factor - построение однофакторных моделей
    PermutationImportance - поочередное перемещивание колонки признака
    """

    def __init__(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                 model: BaseEstimator, metric: str = 'roc_auc', columns: List = None):
        """
        Parameters
        ----------
        X_train:
            Обучающий набор
        X_test:
            Тестовый набор
        y_train:
            Целевая для обучающего набора
        y_test:
            Целевая для тестового набора
        columns:
            Список колонок для анализа
        model:
            Модель, совместимая с sklearn estimator
        """
        if columns is None:
            columns = X_test.columns
        self.X_train = X_train[columns]
        self.X_test = X_test[columns]
        self.y_train = y_train
        self.y_test = y_test
        self.model = model
        self.metric = get_scorer(metric)
        self.metric_name = metric

    def _predict(self, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> float:
        """
        Вывод на экран и получение метрики на тестовом наборе.

        Parameters
        ----------
        X_test:
            Тестовый набор
        y_test:
            Целевая для тестового набора
        threshold:
            Порог для бинаризации

        Return value
        ------------
        Посчитанная метрика
        """

        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = binarize(y_pred_proba.reshape(-1, 1), threshold=threshold).reshape(-1)

        auc = roc_auc_score(y_test.astype(int), y_pred_proba)
        precision = precision_score(y_test.astype(int), y_pred)
        recall = recall_score(y_test.astype(int), y_pred)
        TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

        print('AUC:%.5f, Precision:%.5f, Recall:%.5f, True:%d of %d, Total:%d, Tr:%.2f' %
              (auc, precision, recall,
               TP, TP + FP, TP + FN,
               threshold), flush=True)

        result_metric = self.metric(self.model, X_test, y_test)
        return result_metric

    def _get_metric(self, X_train, X_test, y_train, y_test, threshold=0.5):
        """
        Обучение модели и получение метрики на тестовом наборе.

        Parameters
        ----------
        X_train:
            Тренировочный набор
        X_test:
            Тестовый набор
        y_train:
            Целевая для трени набора
        y_test:
            Целевая для тестового набора
        threshold: float
            Порог для разделения классов
        """

        self.model.fit(X_train, y_train)
        result_metric = self._predict(X_test, y_test, threshold)
        return result_metric

    def _leave_one_out(self, args: List) -> List:
        """
        Метод, который вычисляет AUC без одной фичи. Не должен вызываться напрямую

        Parameters
        ----------
        args:
            Массив из двух значений: feature, full_model_metric
            feature - имя фичи, без которой нужно вычислить метрику
            full_model_metric - метрика модели со всеми фичами

        Return value
        ----------
        [feature, full_model_metric]:
            feature -  имя фичи, без которой вычислен метрика
            full_model_metric - метрика исходной модели - метрика модели без текущего признака.
        """
        feature, full_model_result_metric = args
        result_metric = self._get_metric(self.X_train.drop(feature, axis=1),
                                         self.X_test.drop(feature, axis=1),
                                         self.y_train, self.y_test)
        return [feature, full_model_result_metric - result_metric]

    def get_leave_one_out(self, n_jobs: int = 1, verbose: int = 0) -> pd.DataFrame:
        """
        Подбор признаков Leave-One-Out - поочередное удаление одного из признаков.

        Parameters
        ----------
        n_jobs:
            Количество потоков для вычисления.
        verbose:
            Если значение больше 0, то будет выводиться отладочная информация

        Return value
        ----------
        feature_importance:
            Содержит столбцы [Признак, leave_one_out_*]
                (leave_one_out_auc = метрика исходной модели - метрика модели без текущего признака.)
            отрицательное значение - признак ухудшает модель
            положительное значение - признак улучшает модель
        """
        full_model_result_metric = self._get_metric(self.X_train, self.X_test,
                                                    self.y_train, self.y_test)
        feature_importance = pd.DataFrame()

        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            map(delayed(self._leave_one_out), [(col, full_model_result_metric)
                                               for col in self.X_train.columns]))

        feature_importance = feature_importance.append(results)

        feature_importance.columns = ['features', 'leave_one_out_' + self.metric_name]
        feature_importance = feature_importance.sort_values(by='leave_one_out_' + self.metric_name,
                                                            ascending=False).reset_index(drop=True)
        return feature_importance

    def _one_factor(self, feature: str) -> List:
        """
        Метод, который вычисляет AUC одной фичи. Не должен вызываться напрямую

        Parameters
        ----------
        feature:
            имя фичи, по которой нужно вычислить AUC

        Return value
        ----------
        [feature, metric]:
            feature -  имя фичи, по которой вычислен AUC
            metric - метрика однофакторной модели текущего признака.
        """

        result_metric = self._get_metric(self.X_train[[feature]],
                                         self.X_test[[feature]],
                                         self.y_train, self.y_test)
        return [feature, result_metric]

    def get_one_factor_importance(self, n_jobs: int = 1, verbose: int = 0) -> pd.DataFrame:
        """
        Построение однофакторных моделей - построение моделей от каждого признака.

        Parameters
        ----------
        n_jobs:
            Количество потоков для вычисления.
        verbose:
            Если значение больше 0, то будет выводиться отладочная информация

        Return value
        ----------
        feature_importance: pandas.DataFrame
            Содержит столбцы [Признак, one_fact_*]
                one_fact_* - метрика для однофакторной модели.
        """
        feature_importance = pd.DataFrame()


        results = Parallel(n_jobs=n_jobs, verbose=verbose)(map(delayed(self._one_factor), [col
                                                                                           for col in
                                                                                           self.X_train.columns]))
        feature_importance = feature_importance.append(results)

        feature_importance.columns = ['features', 'one_fact_' + self.metric_name]
        feature_importance = feature_importance.sort_values(by='one_fact_' + self.metric_name,
                                                            ascending=False).reset_index(drop=True)
        return feature_importance


    def _n_permutation_importance(self, args : List ) -> List:
        """
        Метод, который вычисляет метрику для датасета с перемешанным признаком. Не должен вызываться напрямую

        Parameters
        ----------
        model:
            Обученная модель
        feature:
            Имя фичи, которую нужно перемещать для вычисления метрики
        full_model_metric:
            Метрика модели на исходном наборе данных
        n:
            Количество перемешиваний колонки и предсказания на тестовой выборке


        Return value
        ----------
        [feature, metric_name]:
            feature -  имя фичи, по которой вычислена метрика
            metric_name - метрика модели с текущим перемещанным признаком.
        """

        # print(feature+'\n'+full_model_metric)
        feature,  full_model_metric, n = args
        X_test_copy=self.X_test.copy(deep=True)

        scores = []
        for i in range(n):
            np.random.shuffle(X_test_copy[feature].values)
            scores.append(self._predict(X_test_copy, self.y_test))

        result_metric = full_model_metric - np.max(scores) \
            if np.abs(full_model_metric - np.max(scores)) > \
               np.abs(full_model_metric - np.min(scores)) \
            else full_model_metric - np.min(scores)

        return [feature, result_metric]



    def get_n_permutation_importance(self, n: int = 2, model: BaseEstimator = None, n_jobs: int = 1, verbose: int = 0) -> pd.DataFrame:
        """
        Подбор признаков PermutationImportance - поочередная выборка признака,
        его перемещивание и возвращение в датафрейм. Расчет на новом наборе метрики.
    `
        Parameters
        ----------
        n:
            Количество перемешиваний колонки и предсказания на тестовой выборке
        model:
            Бинарный классификатор
            если model = None - обучаем переданную в конструктор класса модель,
            иначе используем предварительно обученную и переданную в данную функцию
        n_jobs:
            Количество потоков для вычисления.
        verbose:
            Если значение больше 0, то будет выводиться отладочная информация

        Return value
        ----------
        feature_importance:
            Содержит столбцы [Признак, permutation_metric_name]
                (метрика = метрика модели на исходном наборе - метрика модели с текущим перемещанным признаком)
            отрицательное значение - признак ухудшает модель
            положительное значение - признак улучшает модель
        """
        if not model:
            self.model = self.model.fit(self.X_train, self.y_train)
        full_model_metric = self._predict(self.X_test, self.y_test)
        feature_importance = pd.DataFrame([])

        for col in self.X_train.columns:
            feature_importance=feature_importance.append([self._n_permutation_importance((col, full_model_metric, n))])

        #results = Parallel(n_jobs=n_jobs, verbose=verbose)(map(delayed(self._n_permutation_importance), [(col, full_model_metric, n)
        #                                                                                   for col in self.X_train.columns]))

        #feature_importance = feature_importance.append(results)

        feature_importance.columns = ['features', 'permutation_' + self.metric_name]

        return feature_importance

    def plot_features_importance(self, features_imp: pd.DataFrame, column_name: str, filename: str = '',
                                 n_features: int = 50, seaborn_palette: str = 'GnBu_d'):
        """
        Построение графика важности признаков.

        Parameters
        ----------
        features_imp:
            Датафрейм: Признак, Метрика
        column_name:
            Название колоки со значениями важности
        filename:
            Имя файла
        seaborn_palette:
            Название seaborn.color_palette для окраски графика
        n_features:
            Количество признаков для вывода на графике
        """
        features_imp = features_imp.sort_values(by=column_name, ascending=False).head(n_features)
        fig = plt.figure(figsize=(8, 8))
        bp = sns.barplot(y=features_imp['features'], x=features_imp[column_name], orient='h',
                         palette=sns.color_palette(seaborn_palette))

        plt.show()

        fig.savefig(filename + '_' + column_name + '.jpg')
