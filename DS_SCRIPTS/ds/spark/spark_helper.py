import os
import sys
import time
from datetime import datetime


class SparkHelper:
    """Класс содержит функции выгрузки и загрузки данных через спарк"""

    def create_sql_context(self, config=None, name='None', port='4440', instances=5, n_cores=2, spark_version='2.1', kernel='PYENV.ZNO0059623792'):
        """
        Создание спарк контекста

        config:
            Конфиг контекста
        name:
            имя процесса
        port:
            spark.ui.port
        instances:
            кол-во instance
        n_cores:
            кол-во cores
        spark_version:
            '2.1', '2.2'
        kernel:
            'PYENV.ZNO20008661'
            'PYENV.ZNO0059623792'

        """
        spark_home = '/opt/cloudera/parcels/SPARK2/lib/spark2'


        os.environ['SPARK_HOME'] = spark_home
        #os.environ['JAVA_HOME'] = '/usr/java/jdk1.8.0_144/'
        os.environ['PYSPARK_DRIVER_PYTHON'] = 'python'
        sys.path.insert(0, os.path.join (spark_home,'python'))
        sys.path.insert(0, os.path.join (spark_home,'python/lib/py4j-0.10.7-src.zip'))
        os.environ['SPARK_HOME'] = spark_home
        os.environ['LD_LIBRARY_PATH'] =  '/opt/python/virtualenv/jupyter/lib'
        os.environ['PYHTON_PATH'] = '/opt/workspace/libs/'

        from pyspark import SparkContext, SparkConf, HiveContext

        if config is None:
            config = SparkConf().setMaster("yarn-client").setAppName(name+'_'+port)
            config.setAll(
                [
                   ('spark.local.dir', 'sparktmp'),
                   ('spark.executor.memory','15g'),
                   ('spark.driver.maxResultSize','80g'),
                   ('spark.executor.instances', instances),
                   ('spark.yarn.driver.memoryOverhead', '3024mb'),
                   ('spark.port.maxRetries', '500'),
                   ('spark.executor.cores', n_cores),
                   ('spark.dynamicAllocation.enabled', 'false'),
                   ('spark.kryoserializer.buffer.max.mb','1g'),
                   ('spark.ui.port', port),
                   ('spark.blacklist.enabled', 'true'),
                   ('spark.sql.shuffle.partitions', '400'),
                   ('spark.blacklist.task.maxTaskAttemptsPerNode', '15'),
                   ('spark.blacklist.task.maxTaskAttemptsPerExecutor', '15'),
                   ('spark.task.maxFailures', '16'),
                   ('hive.metastore.uris', 'thrift://februs6.lab.df.sbrf.ru:9083'),
                   ('hive.metastore.uris', 'thrift://februs7.lab.df.sbrf.ru:9083')
                ])

        print('Start')
        self.sc = SparkContext.getOrCreate(conf=config)
        self.sqlContext = HiveContext(self.sc)
        print('Context ready: %s' % self.sc)

        return self.sc, self.sqlContext


    def save_to_csv(self, df, sep:str, username:str,  hdfs_path:str, local_path:str=None, isHeader='true'):
        """
        Сохраняет Spark DataFrame с csv и создает линк на этот файл в файловой системе Jupyter

        Parameters
        ----------
        username:
            Имя пользователя в ЛД
        hdfs_path:
            Путь для сохранения файла в HDFS относительно папки пользователя (например notebooks/data)
        local_path:
            Путь, по которому будет доступен файл в файловой системе Jupyter (/home)
            Если None - запись производится только в hdfs
        """
        import subprocess
        import os

        df.write \
            .format('com.databricks.spark.csv') \
            .mode('overwrite') \
            .option('sep', sep) \
            .option('header', isHeader) \
            .option("quote", '\u0000') \
            .save(hdfs_path)

        if local_path!=None:
            path_to_hdfs = os.path.join('/user', username, hdfs_path)
            path_to_local = os.path.join('/home', username, local_path)
            proc = subprocess.Popen(['hdfs', 'dfs', '-getmerge', path_to_hdfs, path_to_local])
            proc.communicate()

            columns_row = sep.join(df.columns)
            os.system("sed -i -e 1i'" + columns_row + "\\' " + path_to_local)


    def read_from_csv(self, hdfs_path:str, schema=None):
        """
        Чтения из csv в Spark Dataframe
        Parameters
        ----------
        hdfs_path:
            Путь файла в HDFS относительно папки пользователя (например notebooks/data)
        schema:
            Схема таблицы
        """

        if schema is None:
            return sqlContext.read.format('com.databricks.spark.csv') \
            .option('inferSchema', 'true') \
            .option('header', 'true') \
            .option('delimiter', ';') \
            .option('decimal', '.') \
            .option('dateFormat', 'yyyy-MM-dd') \
            .option('encoding', 'cp1251') \
            .load(hdfs_path)
        else:
            return sqlContext.read.format('com.databricks.spark.csv') \
                .schema(schema) \
                .option('header', 'true') \
                .option('delimiter', ';') \
                .option('decimal', '.') \
                .option('dateFormat', 'yyyy-MM-dd') \
                .option('encoding', 'cp1251') \
                .load(hdfs_path)


    def create_table_from_select(self, db_in='t_ural_kb', table_name='', db_out='t_ural_kb'):
        """
        Для копирования таблиц из схемы в схему

        Parameters
        ----------

        db_in:
             схема из которой нужно копировать
        table:
             копируемая таблица
        db_out:
             схема в которую копируем
        """

        start=time.time()
        print("Start copying in {db_in}.{table} to {db_out}.'0_'+{table}".format(db_in=db_in, table=table_name, db_out=db_out))

        self.sqlContext.sql('DROP TABLE IF EXISTS {db_out}.{tab}'.format(db_out=db_out, tab=table_name))
        sql_query="""create table {db_out}.{table}
                            as
                            select *
                            from {db_in}.{table}""".format(db_in=db_in, table=table_name, db_out=db_out)

        self.sqlContext.sql(sql_query)

        print("End copying in {db_in}.{table} to {db_out}.'0_'+{table} ...  {t:0.2f} seconds ".format(db_in=db_in, table='0_'+table_name, db_out=db_out, t=(start-time.time())))


    def save_to_hive(self, df, db='t_ural_kb', table_name='temp_{d:}'.format(d=datetime.now().date()).replace('-', '_')):
        """
        Сохранение spark датафрейма в hive
        Parameters
        ----------
        df:
            spark датафрейм для записи в таблицу
        db:
            схема в которой создается таблица
        table_name:
            название таблицы, если не указано temp_{текущая дата} (temp_01_07_2019)
        """
        start=time.time()
        df.registerTempTable(table_name)
        self.sqlContext.sql('DROP TABLE IF EXISTS {db}.{tab}'.format(db=db, tab=table_name))
        self.sqlContext.sql('CREATE TABLE {db}.{tab} SELECT * FROM {tab}'.format(db=db, tab=table_name))
        print("End creating {db}.{table}...  {t:0.2f} seconds ".format(db=db, table=table_name, t=(time.time()-start)))


    def get_table (self, db, table_name, columns=[]):
        """
        Селект таблицы
        Parameters
        ----------
        db:
            схема из которой селектится таблица
        table_name:
            название таблицы
        columns:
            список колонок для селекта, если пустой *
        """
        return self.sqlContext.sql('SELECT {cols} FROM {db}.{tab}'.format(db=db, tab=table_name, cols=', '.join(columns) if columns!=[] else '*'))

    def add_prefix_col_name(self, df, prefix):
        """
        Добавление префикса ко всем названиям колонок в таблице
        Parameters
        ----------
        df:
            датафрейм в котором нужно переименовать столбцы
        prefix:
            префикс для столбцов

        """
        for col in df.columns:
            df=df.withColumnRenamed(col, prefix+'_'+col)
        return df

    def cast_columns(self, df, columns_types):
        """
        Приведение типов к списку колонок
        Parameters
        ----------
        df:
            спарк датафрейм
        columns_types:
            словарь соответствий колонка : тип к которому нужно привести колонку
        """
        from pyspark.sql import functions  as F
        for col_name, type_col in columns_types.items():
            df=df.withColumn(col_name, F.col(col_name).cast(type_col))

        return df

