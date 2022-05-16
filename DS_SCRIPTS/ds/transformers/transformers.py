import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnsSelector(BaseEstimator, TransformerMixin):
    """
        Selects columns from dataframe
    """

    def __init__(self, columns=[]):
        self.columns = columns

    def transform(self, X):
        """
            Selects specified by self.columns columns from dataframe

            Args:
                X (pandas.DataFrame): dataframe

            Returns:
                pandas.DataFrame: selected columns
        """
        selected_columns = X[self.columns].copy()
        return selected_columns

    def fit(self, X, y=None):
        return self


class TypeSelector(BaseEstimator, TransformerMixin):
    """
        Selects columns of certain type from dataframe
    """
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
            Selects specified by self.dtype columns from dataframe

            Args:
                X (pandas.DataFrame): dataframe

            Returns:
                pandas.DataFrame: selected columns with type=dtype
        """
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


class Woe(BaseEstimator, TransformerMixin):
    """
        Performs woe transformation of selected columns
    """
    def __init__(self, num_bins):
        self.num_bins = num_bins

    def fit(self, X, y):
        """
            Performs woe transformation for specified by self.columns columns
            from dataframe and save dictionary of woes for categories or bins

            Args:
                X (pandas.DataFrame): dataframe
                y (pandas.Series): target
        """
        assert X.shape[0] == y.shape[0],\
         'Shapes of data and target must be the same. X shape:{0}, y shape:{1}'.format(X.shape[0], y.shape[0])

        self.woes = dict.fromkeys(X.columns)
        self.bins = dict.fromkeys(X.columns)

        for col_name in X.columns:
            # Concat column to be transformed and target
            df = pd.concat([X[col_name], y], axis=1)
            df.columns = [col_name, 'target']

            if str(df[col_name].dtype) == 'category':
                df[col_name] = df[col_name].cat.add_categories([-1])
            df[col_name].fillna(-1, inplace=True)

            # Save overall number of events(target=1) and non-events(target=0)
            events_num = df[df['target'] == 1].shape[0]
            non_events_num = df[df['target'] == 0].shape[0]

            # Check column type and define bins
            if X[col_name].dtype == np.int or X[col_name].dtype == np.float:
                series, bins = pd.qcut(X[col_name], q=self.num_bins, duplicates='drop', retbins=True)
                self.bins[col_name] = bins
            else:
                series = col_name
                self.bins[col_name] = X[col_name].unique()

            # print(series)
            # Group by category or bin and
            # count number of events(1s) and non-events(0s) for every group
            gb = df.groupby(series)['target'].agg(['count', 'sum'])
            gb.columns = ['group_count', 'events']
            gb['non_events'] = gb['group_count'] - gb['events']

            # Count % events and % non-events
            # with respect to overall number of events and non-events
            events_percent = gb['events'] / events_num
            non_events_percent = gb['non_events'] / non_events_num

            # Save woes for column in dictionary
            self.woes[col_name] = np.log1p(non_events_percent / (events_percent + 1e-3))

        return self

    def transform(self, X):
        """
            Applies woe transformation specified by self.columns columns from dataframe

            Args:
                X (pandas.DataFrame): dataframe

            Returns:
                pandas.DataFrame: woe columns
        """
        transformed_cols = pd.DataFrame()
        for key, value in self.woes.items():
            # If column is int or float then first get binned column
            if X[key].dtype == np.int or X[key].dtype == np.float:
                series = pd.cut(X[key], bins=self.bins[key], include_lowest=True)
            else:
                series = X[key]

            # print(self.bins[key])
            # Map woe for categories or bins. Fill with mean woe
            # if category didn't encounter during fit
            transformed_cols['woe_' + key] =  np.nan_to_num(series.map(value).fillna(value.mean()))
        return transformed_cols
