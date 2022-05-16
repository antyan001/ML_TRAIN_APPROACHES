import pandas as pd
import numpy as np
import seaborn as sns
import time
import os
import sys
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, \
                            classification_report, precision_score, \
                            recall_score, roc_curve,confusion_matrix,\
                            precision_recall_curve, average_precision_score

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import binarize
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin

__all__ = ('TrainNFolds')

class TrainNFolds(BaseEstimator, ClassifierMixin):
     '''
    В классе реализованы метод скользящего по n-подмножествам
    Пример:
    На первой итерации обучение производится на первом и втором из фолдов, и тестирование на третьем -  производится подсчет метрик,
    на следующих итерациях роль фолдов циклично менялась.
    Таким образом, модель была обучается на трех разных подвыборках, а результативное предсказание является усреднением трех моделей.
    '''
     def __init__(self, model, n_folds=3, models_path='models',threshold=0.5, fit_params={}):
        '''
        Parameters
        ----------
        n_folds:  int
            Количество подмножеств
        models_path: string
            Папка в которой хранятся модели
        model:
             Алгоритм машинного обучения sklearn, xgboost, и.т.д.
        threshold: float
            Порог, отделяющий 1 и 0 классы
        fit_params: dictionary
            Словарь с параметрами для обучения
        '''
        self.models_path=models_path
        self.n_folds = n_folds
        self.model=model
        self.threshold=threshold
        self.fit_params=fit_params

     def _create_dirictory(self, name):
        '''
        Построение кривых полноты и точности, вывод точки их пересечения
        Parameters
        ----------
        name: string
            Название папки\путь
        '''
        if not os.path.exists(name):
            os.mkdir(name)

     def _get_precision_recall_curves(self, y, y_score):
        '''
        Построение кривых полноты и точности, вывод точки их пересечения
        Parameters
        ----------
        y: pandas.Series
            Целевая переменная
        y_score: pandas.Series
            Предсказание из predic_proba
        '''
        recalls    = []
        precisions = []
        thresholds = np.arange(0.01, 1.0, 0.01)
        for t in thresholds:
            pred = binarize(y_score.reshape(-1,1), threshold=t).reshape(-1)
            recall = recall_score(y , pred)
            precision=precision_score(y, pred)
            recalls.append(recall)
            precisions.append(precision)
        print('Cross point: ', np.argwhere(np.isclose(recalls, precisions, atol=0.02)).ravel()[0])
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        title = 'Точность и полнота, в зависимости от порога'
        plt.plot(thresholds, precisions, label='precision')
        plt.plot(thresholds, recalls, label='recall')
        plt.xlabel('threshold')
        plt.ylabel('precision/recall')
        plt.legend(loc='best')
        plt.title(title)
        plt.show()

     def _eval_model(self, X, y):
        '''
        Parameters
        Вывод метрик
        ----------
        __X: pandas.DataFrame
            Обучающий набор
        __y: pandas.Series
            Целевая для обучающего набора
        '''
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = binarize(y_pred_proba.reshape(-1,1), threshold=self.threshold).reshape(-1).astype(int)

        print('AUC:%.5f, Precision:%.5f, Recall:%.5f, F1-score:%.5f, Tr:%.2f' %
              (roc_auc_score(y, y_pred_proba),
               precision_score(y, y_pred),
               recall_score(y, y_pred),
               f1_score(y, y_pred),
               self.threshold), flush=True)

        return y_pred, y_pred_proba

     def fit(self, X, y=None):
        '''
        Обучение модели
        Parameters
        ----------
        X: pandas.DataFrame
            Обучающий набор
        y: pandas.Series
            Целевая переменная для обучающего набора
        '''
        self._create_dirictory(self.models_path)
        blend_train       = np.zeros(X.shape[0])
        blend_train_proba = np.zeros(X.shape[0])

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        if str(type(X))=="<class 'pandas.core.frame.DataFrame'>":
            X=X.values
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.model.fit(X_train, y_train, **self.fit_params)
            blend_train[test_index], blend_train_proba[test_index] = self._eval_model(X_test, y_test)
            joblib.dump(self.model, self.models_path+'/'+str(i)+'.pkl', compress=9)

        print('-' * 68)
        print(classification_report(y, blend_train))
        print('AUC: %7.5f' % (roc_auc_score(y, blend_train_proba)), flush=True)

        self._get_precision_recall_curves(y, blend_train_proba)
        self.classes_=list(set(y))

     def predict_proba(self, X):
        '''
        Усредненное по фолдам предсказание модели

        Parameters
        ----------
        X: pandas
            Набор данных
        '''
        if str(type(X))=="<class 'pandas.core.frame.DataFrame'>":
            X=X.values
        predicted=np.zeros([X.shape[0], 2])
        for i in range(0, self.n_folds):
            clf=joblib.load(self.models_path+'/'+str(i)+'.pkl')
            predicted+=clf.predict_proba(X)

        predicted/=self.n_folds
        return predicted

     def predict(self, X, y=None):
        predicted=self.predict_proba(X)[:, 1]
        return binarize(predicted.reshape(-1,1), threshold=self.threshold).reshape(-1).astype(int)

     def test(self, X , y, threshold=0.5):
        '''
        Тестирование модели

        Parameters
        ----------
        X: pandas
            Обучающий набор
        y: pandas
            Целевая для обучающего набора
        threshold: float
            Порог, отделяющий 1 и 0 классы
        models_path:  string
            Папка из которой загружаются модели
        '''

        predicted_proba=self.predict_proba(X)[:, 1]
        # predicted['PREDICTION']=0
        predicted=binarize(predicted_proba.reshape(-1,1), threshold=threshold).reshape(-1).astype(int)

        print('AUC: %7.5f, Precision: %7.5f, Recall: %7.5f, f1-score: %7.5f' %
              (roc_auc_score(y, predicted_proba),
               precision_score(y, predicted),
               recall_score(y, predicted),
               f1_score(y, predicted)
              ), flush=True)

        print(classification_report(y, predicted))

        return predicted_proba


