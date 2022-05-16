from sklearn.preprocessing import LabelEncoder
import pandas as pd

__all__ = ('SoftLabelEncoder')

class SoftLabelEncoder():
    __dictionary = {}
    __max_value = 0

    def __init__(self, sklearn_label_encoder=None):
        '''
        LabelEncoder из sklearn с возможностью преобразовывать колонки с новыми значениями.

        Parameters
        ----------
        sklearn_label_encoder: object
            Обученный LabelEncoder из Sklearn
        '''
        if sklearn_label_encoder is not None:
            self.__max_value = sklearn_label_encoder.transform(sklearn_label_encoder.classes_).max()
            self.__dictionary = dict(zip(sklearn_label_encoder.classes_, \
                                        sklearn_label_encoder.transform(sklearn_label_encoder.classes_)))

    def __import_sklearn_label_encoder(self, le):
        self.__max_value = le.transform(le.classes_).max()
        self.__dictionary = dict(zip(le.classes_, le.transform(le.classes_)))

    def fit(self, y):
        """Fit label encoder

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        """
        le = LabelEncoder()
        le.fit(y)
        self.__import_sklearn_label_encoder(le)

    def transform(self, y):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like or pandas Series of shape [n_samples]
            Target values.

        Returns
        -------
        y : pandas Series of shape [n_samples]
        """
        new_values = set(y) - set(self.__dictionary.keys())
        if len(new_values) > 0 and self.__max_value != 0:
            print('New labels: ', ', '.join(new_values))
        for val in new_values:
            self.__dictionary[val] = self.__max_value + 1
            self.__max_value += 1
        if type(y) == list:
            y = list(pd.Series(y).map(self.__dictionary))
        else:
            y = y.map(self.__dictionary)
        return y

    def inverse_transform(self, y):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : numpy array or pandas Series of shape [n_samples]
            Target values.

        Returns
        -------
        y : pandas Series of shape [n_samples]
        """
        new_values = set(y) - set(self.__dictionary.values())
        if new_values:
            raise ValueError('y contains new labels: %s' % str(new_values))
        inversed_dict = dict(zip(self.__dictionary.values(), self.__dictionary.keys()))
        if type(y) == list:
            y = list(pd.Series(y).map(inversed_dict))
        else:
            y = y.map(inversed_dict)
        return y

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels

        Parameters
        ----------
        y : array-like or pandas Series of shape [n_samples]
            Target values.

        Returns
        -------
        y : pandas Series of shape [n_samples]
        """
        self.fit(y)
        return self.transform(y)

    def get_dict(self):
        """
        Returns
        -------
        dictionary : dict
            Dictionary, that uses for transformation
        """
        return self.__dictionary.copy()

    def get_labels(self):
        """
        Returns
        -------
        values : list
            Int values, that assigned for lables.
        """
        return self.__dictionary.values()

    def get_keys(self):
        """
        Returns
        -------
        labels : list
            String values, lables.
        """
        return self.__dictionary.keys()
