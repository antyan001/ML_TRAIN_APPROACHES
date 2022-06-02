# -*- coding: utf-8 -*-
import cx_Oracle
import pandas as pd
import time
import datetime
import os

os.environ['NLS_LANG'] = 'RUSSIAN_RUSSIA.AL32UTF8'

class OracleDB(object):

    def __init__(self, user, password, sid, init_tns=True):
        self.user = user
        self.password = password
        self.sid = sid
        self.DatabaseError = cx_Oracle.DatabaseError
        self.tns_names = {
            'iskra4': cx_Oracle.makedsn('',  1521, 'iskra4'),
            'iskra3': cx_Oracle.makedsn('', 1521, 'iskra3'),
            'iskra2': cx_Oracle.makedsn('', 1521, 'iskra2'),
            'iskra1_primary': """(DESCRIPTION =
               (LOAD_BALANCE=off)
                 (FAILOVER = ON)
                 (ADDRESS_LIST =
                      (ADDRESS = (PROTOCOL = TCP)  (HOST = iskra10.ca.sbrf.ru) (PORT = 1521) )
                      (ADDRESS = (PROTOCOL = TCP)  (HOST = iskra11.ca.sbrf.ru) (PORT = 1521) )
                      )
                 (CONNECT_DATA = (SERVICE_NAME = cskostat_primary) (FAILOVER_MODE= (TYPE=select) (METHOD=basic)
                      )
                     ))"""
             }
        self.init_tns = init_tns


    def connect(self):
        if self.init_tns:
            self.connection = cx_Oracle.connect(self.user, self.password, self.tns_names[self.sid])
        else:
            self.connection = cx_Oracle.connect('{0}/{1}@{2}'.format(self.user, self.password, self.sid))
        self.cursor = self.connection.cursor()

    def close(self):
        self.cursor.close()
        self.connection.close()


class Loader(object):
    """
    Class contains functions for loading data from database.

    Functions:
        get_dataframe - return dataframe
        save_csv - save csv to particular path
    """

    def __init__(self, iskra, password, init_tns=True):
        self.login =  iskra
        self.password = password
        self.init_tns = init_tns

    def get_dataframe(self, query):
        """
        Return dataframe  for specified query.

        Args:
            query (str): sql query
            iskra (str): from which iskra get table
        Returns:
            pandas.DataFrame: dataframe  for specified query
        """
        try:
            db = OracleDB('iskra', self.password, self.login, self.init_tns)
            db.connect()
            df = pd.read_sql(query, con=db.connection)
            df.columns = map(str.lower, df.columns)
            return df
        except Exception as e:
            print(str(e))

    def _read_from_db(self, db, columns, file_name, append_header):
        rows = db.cursor.fetchmany()
        if not rows:
            return False

        data = pd.DataFrame(rows, columns=columns)
        data.to_csv(file_name, mode='a', index=False,
                    header=append_header, sep=';',  encoding='utf8')
        append_header = False

        return True

    def _get_balance(self, db, sql, file_name, verbose):
        print('Connecting...')
        db.connect()

        start_time = datetime.datetime.now()
        print('Getting data ... ')
        db.cursor.arraysize = 10000

        db.cursor.execute(sql)
        columns = [x[0].lower() for x in db.cursor.description]

        start_time = time.time()
        append_header = True

        if os.path.exists(file_name):
            os.remove(file_name)

        count = 0
        while True:
            result = self._read_from_db(db, columns, file_name, append_header)
            count += 1
            append_header = False
            if not result:
                break
            if verbose == 1:
                print('Downloaded {:,} lines, {:7.0f}sec. passed'
                      .format(count * db.cursor.arraysize,
                              time.time() - start_time))
        db.close()

    def save_csv(self, query, path='data.csv', verbose=1):
        """
        Saves csv  for specified query.

        Args:
            query (str): sql query
            path (str): path
            iskra (str): from which iskra get table
        """
        try:
            db = OracleDB('iskra', self.password, self.login, self.init_tns)
            self._get_balance(db, query, path, verbose)
            db.close()
        except Exception as e:
            print(str(e))

    def insert(self, df, sql_insert):
        """
        Insert into Oracle table using executemany function.
        Order of df must be the same as in sql_insert.

        Example of sql_insert string:
        insert into exmaple_table (id, value1, value2)
                         values (:1 , :2, :3)

        Args:
            df (pd.DataFrame): dataframe to insert
            sql_insert (str): SQL code with isert
            iskra (str): from which iskra get table
        """
        rows = [tuple(x) for x in df.values]

        db = OracleDB('iskra', self.password, self.login, self.init_tns)
        db.connect()
        cur = db.cursor
        cur.executemany(sql_insert, rows)
        cur.execute('commit')
        db.close()

    def get_connection(self):
        """
        Return open connection to database.
        Remember to close the connection after use!

        """
        db = OracleDB('iskra', self.password, self.login, self.init_tns)
        db.connect()
        return db.connection
