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


__all__ = ('TrainNFolds')

class TrainNFolds:
     '''
    В классе реализованы метод скользящего по n-подмножествам
    Пример:
    На первой итерации обучение производится на первом и втором из фолдов, и тестирование на третьем -  производится подсчет метрик,
    на следующих итерациях роль фолдов циклично менялась.
    Таким образом, модель была обучается на трех разных подвыборках, а результативное предсказание является усреднением трех моделей.
    '''
     def __init__(self, n_folds=3, models_path='models'):
        '''
        Parameters
        ----------
        n_folds:  int
            Количество подмножеств
        models_path: string
            Папка в которой хранятся модели
        '''
        self.models_path=models_path
        self.n_folds = n_folds

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

     def _eval_model(self, model, X, y, threshold=0.5):
        '''
        Parameters
        Вывод метрик
        ----------
        __X: pandas.DataFrame
            Обучающий набор
        __y: pandas.Series
            Целевая для обучающего набора
        n_folds:  int
            Количество подмножеств
        '''
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = binarize(y_pred_proba.reshape(-1,1), threshold=threshold).reshape(-1).astype(int)

        print('AUC:%.5f, Precision:%.5f, Recall:%.5f, F1-score:%.5f, Tr:%.2f' %
              (roc_auc_score(y, y_pred_proba),
               precision_score(y, y_pred),
               recall_score(y, y_pred),
               f1_score(y, y_pred),
               threshold), flush=True)

        return y_pred, y_pred_proba

     def train(self, model, X, y, threshold=0.5, fit_params={}):
        '''
        Обучение модели
        Parameters
        ----------
        model:
             Алгоритм машинного обучения sklearn, xgboost, и.т.д.
        X: pandas.DataFrame
            Обучающий набор
        y: pandas.Series
            Целевая переменная для обучающего набора
        threshold: float
            Порог, отделяющий 1 и 0 классы
        '''
        self._create_dirictory(self.models_path)
        blend_train       = np.zeros(X.shape[0])
        blend_train_proba = np.zeros(X.shape[0])

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train, **fit_params)
            blend_train[test_index], blend_train_proba[test_index] = self._eval_model(model, X_test, y_test, threshold=threshold)
            joblib.dump(model, self.models_path+'/'+str(i)+'.pkl', compress=9)

        print('-' * 68)
        print(classification_report(y, blend_train))
        print('AUC: %7.5f' % (roc_auc_score(y, blend_train_proba)), flush=True)

        self._get_precision_recall_curves(y, blend_train_proba)

     def prediction(self, X):
        '''
        Усредненное по фолдам предсказание модели

        Parameters
        ----------
        X: pandas
            Набор данных
        '''
        predicted=pd.DataFrame(index=X.index)
        predicted['PROBABILITY']=0
        for i in range(0, self.n_folds):
            clf=joblib.load(self.models_path+'/'+str(i)+'.pkl')
            predicted['PROBABILITY']+=clf.predict_proba(X)[:, 1]

        predicted['PROBABILITY']/=self.n_folds
        return predicted


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

        predicted=self.prediction(X)
        # predicted['PREDICTION']=0
        predicted['PREDICTION']=binarize(predicted['PROBABILITY'].values.reshape(-1,1), threshold=threshold).reshape(-1).astype(int)

        print('AUC: %7.5f, Precision: %7.5f, Recall: %7.5f, f1-score: %7.5f' %
              (roc_auc_score(y, predicted['PROBABILITY']),
               precision_score(y, predicted['PREDICTION']),
               recall_score(y, predicted['PREDICTION']),
               f1_score(y, predicted['PREDICTION'])
              ), flush=True)

        print(classification_report(y, predicted['PREDICTION']))

        return predicted


