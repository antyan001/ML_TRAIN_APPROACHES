 * SparkCsv - класс для загрузки и чтения csv в Spark
Разработчик Ватолин Алексей

 * Plot - класс для визуализации метрик и feature_importance
Разработчик Ватолин Алексей


 * feature_importance - класс для анализа важности признаков (leave-one-out, permutation, one-factor)
Разработчик Патракова Екатерина

* train_algoritms/Compare+CatBoost+XGBoost+and+LightGBM.ipynb - обучение трех моделей (XGBoost, CatBoost и LightGBM) и сравнение метрик
Разработчик Ватолин Алексей

* feature_engeniring/label_encoder - класс для кодирования текстовых значений натуральными числами. Отличается от LabelEncoder из sklearn тем, что не падает при появлении новых значений в трансформируемых данных.
Разработчик Ватолин Алексей

* load_data/loader.py - класс для загрузки данных из таблицы и сохранения в файл. Такой же код, что и в Download to csv VIA Cursor+memory clean.ipynb, но оформленный в класс.
Разработчик Александрин Виктор


* feature_engeniring/forward_permutationFE - отбор фичей с помощью permutation важности. Фичи добавляются по одной.
Разработчик Ватолин Алексей

* feature_engeniring/forward_permutationFE_cv - отбор фичей с помощью permutation важности. Фичи добавляются по одной и затем вычисляется увеличение метрики на кроссвалидации
Разработчик Ватолин Алексей

* feature_engeniring/ReduceVIF - отбор фичей удаляя те, которые с хорошей точностью являются линейной комбинацией остальных. Метод VIF.
Разработчик индус с Kaggle, поддержка Ватолин Алексей

* train_algoritms/train_folds - обучение, тестирование и предсказание модели по фолдам.
Разработчик Патракова Екатерина
