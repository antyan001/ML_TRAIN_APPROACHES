from sklearn.metrics import confusion_matrix, average_precision_score,  \
    roc_auc_score, precision_recall_curve, roc_curve
from sklearn.preprocessing import binarize
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

__all__ = ('Plot')

class Plot():
    
    @classmethod
    def confusion_matrix_heatmap(self, y_test, y_pred):
        '''
        Построение Confusion matrix (матрицы ошибок)

        Parameters
        ----------
        y_test: pandas.Series, numpy.array
            Целевая для обучающего набора
        y_pred: pandas.Series, numpy.array
            Значения целевой переменной, предсказанные классификатором
        '''
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    @classmethod 
    def feature_impotance(self, model, X_train, count_top_features = 20, palette_name = 'YlGn'):
        '''
        Построение графика важности фич

        Parameters
        ----------
        model: object
            Объект модели
        X_train: pandas.Series
            Колонка данных для обучения для извлечения названий фич
        count_top_features: int
            Количество выводимых фич
        palette_name: str
            Имя палитры Seaborn
        '''
        imp=model.feature_importances_
        names=X_train.columns
        imp, names=map(list, zip(*sorted(zip(imp, names))[::-1][:count_top_features]))

        ax = plt.axes()
        sns.barplot(x=imp, y=names, palette=sns.color_palette(palette_name, 2), ax=ax)
        ax.set_title('Top ' + str(count_top_features) + ' important features')
        plt.show()

    @classmethod
    def precision_recall_curve(self, y_test, y_test_proba):
        '''
        Построение кривой зависимости precision от recall.
        По оси Х recall, по у - precision

        Parameters
        ----------
        y_test: pandas.Series, numpy.array
            Целевая для обучающего набора
        y_test_proba: pandas.Series, numpy.array
            Вероятности целевой переменной, предсказанные классификатором.
            Пример: xgb.predict_proba(X_test)[:,1]
        '''
        average_precision = average_precision_score(y_test, y_test_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
        fig = plt.figure(figsize=(9,7))
        plt.step(recall, precision, color='b', where='post', alpha=0.2)
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0, 1.05])
        plt.xlim([0, 1])
        plt.title('2-class Precision-Recall curve: Average precision = %.3f' % average_precision)
        plt.show()

    @classmethod
    def auc_curve(self, y_test, y_test_proba):
        '''
        Построение AUC кривой

        Parameters
        ----------
        y_test: pandas.Series, numpy.array
            Целевая для обучающего набора
        y_test_proba: pandas.Series, numpy.array
            Вероятности целевой переменной, предсказанные классификатором.
            Пример: xgb.predict_proba(X_test)[:,1]
        '''
        fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
        AUC = roc_auc_score(y_test, y_test_proba)
        fig = plt.figure(figsize=(9,7))
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % AUC)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

    @classmethod
    def precision_recall_threshold_curve(self, y_true, y_test_proba):
        '''
        Построение кривых зависимости precision и recall от threshold.

        Parameters
        ----------
        y_test: pandas.Series, numpy.array
            Целевая для обучающего набора
        y_test_proba: pandas.Series, numpy.array
            Вероятности целевой переменной, предсказанные классификатором.
            Пример: xgb.predict_proba(X_test)[:,1]
        '''
        recalls    = []
        precisions = []
        thresholds = np.arange(0.01, 1.05, 0.05)
        for t in thresholds:
            pred = binarize(y_test_proba.reshape(-1,1), threshold=t).reshape(-1)
            tp = np.sum(np.logical_and(pred == 1, y_true == 1))
            tn = np.sum(np.logical_and(pred == 0, y_true == 0))
            fp = np.sum(np.logical_and(pred == 1, y_true == 0))
            fn = np.sum(np.logical_and(pred == 0, y_true == 1))
            recall    = tp / (tp + fn)
            precision = tp / (tp + fp)
            recalls.append(recall)
            precisions.append(precision)
        #print('Cross point: ', '%0.3f' % precisions[np.argwhere(np.isclose(recalls, precisions, atol=0.01)).ravel()[0]])
        fig = plt.figure(figsize=(9,5))
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        title = 'Precision and F1 depending on the threshold'
        plt.plot(thresholds, precisions, label='precision')
        plt.plot(thresholds, recalls, label='recall')
        plt.xlabel('threshold')
        plt.ylabel('precision/recall')
        plt.legend(loc='best')
        plt.title(title)
        plt.show()

    @classmethod
    def distplot(self, y_test_proba): 
        fig = plt.figure(figsize=(9,5))
        ax = plt.axes()
        title = 'Distribution of predicted probabilities'
        sns.distplot(y_test_proba, axlabel='Probabilities from estimator',
                                    kde_kws={'color': 'r', 'lw': 2, 'label': 'KDE'})
        plt.title(title)
        plt.show()
        
        