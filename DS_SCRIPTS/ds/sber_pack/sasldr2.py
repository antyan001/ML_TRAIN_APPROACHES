import cx_Oracle
from os.path import exists
import win32com.client as wcc
import datetime
import os
import time
import pytz
from sber_pack.sber_lib import get_iskra_con_str, oracle_logging, open_imp_log, update_imp_log, close_imp_log, close_error_imp_log
from sber_pack.tasks_api import *
import re

def StartOper(cnctn, type_oper_id):
    try:
        con=cx_Oracle.connect(cnctn)
        cur=con.cursor()
        opid=cur.callfunc("LG_PKG.StartOper",int,[type_oper_id])
        cur.close()
        con.close()
        return opid
    except:
        return None

def RegPhase(cnctn, opid, name):
    try:
        con=cx_Oracle.connect(cnctn)
        cur=con.cursor()
        stid=cur.callfunc("LG_PKG.RegPhase",int,[opid, name])
        cur.close()
        con.close()
        return stid
    except:
        return None 

def EndOper(cnctn, opid):
    try:
        con=cx_Oracle.connect(cnctn)
        cur=con.cursor()
        cur.callproc("LG_PKG.EndOper",[opid])
        cur.close()
        con.close()
    except:
        pass

def EndPhase(cnctn, stid):
    try:
        con=cx_Oracle.connect(cnctn)
        cur=con.cursor()
        cur.callproc("LG_PKG.EndPhase",[stid])
        cur.close()
        con.close()
    except:
        pass

def EndPhaseWithRows(cnctn, stid, t_cnt):
    try:
        con=cx_Oracle.connect(cnctn)
        cur=con.cursor()
        cur.callproc("LG_PKG.EndPhase",[stid, t_cnt, 0, 0, 0])
        cur.close()
        con.close()
    except:
        pass

def SetParam(cnctn, opid, param_name, param_value, param_type, stid=None):
    try:
        con=cx_Oracle.connect(cnctn)
        cur=con.cursor()
        cur.callproc("LG_PKG.SetParam",[opid, param_name, param_value, param_type, stid])
        cur.close()
        con.close()
    except:
        pass

def AddLog(cnctn, stid, log_type, log_text):
    try:
        con=cx_Oracle.connect(cnctn)
        cur=con.cursor()
        cur.callproc("LG_PKG.AddLog",[stid, log_type, log_text])
        cur.close()
        con.close()
    except:
        pass

def sasldr_logging(path,message):
    print(message)
    with open(path, 'a') as l:
        l.write('{} {}\n'.format(str(datetime.datetime.now()),message))
        l.close()

def need_update(connection, name):
    con=cx_Oracle.connect(connection)
    cur=con.cursor()
    res=cur.execute("""select * from td_loadings where name = '{}'""".format(name)).fetchone()
    cur.close()
    con.close()
    if res[2]=='UPDATED' and res[1].date()==datetime.datetime.now(pytz.timezone('Europe/Moscow')).date():
        return False
    else:
        return True

def td_iskra_log(connection,name,status,error=''):
    con=cx_Oracle.connect(connection)
    cur=con.cursor()
    dttm=datetime.datetime.now(pytz.timezone('Europe/Moscow')).strftime('%d.%m.%Y %H:%M:%S')
    cur.execute("""update td_loadings set dttm=to_date('{}','dd.mm.yyyy HH24:MI:SS'), status='{}' , error = '{}' where name = '{}'""".format(dttm,status,error,name))
    con.commit()
    cur.close()
    con.close()

def log_parser(log_path):
    if not exists(log_path):
        return []
    with open(log_path,'r') as f:
        log=f.read()
        f.close()
    notes=log.split('NOTE:')
    if len(notes) == 1:
        if len(log)==len(notes[0]):
            notes=[]
    print('{} notes'.format(len(notes)))
    warnings=log.split('WARNING:')
    if len(warnings) == 1:
        if len(log)==len(warnings[0]):
            warnings=[]
    print('{} warnings'.format(len(warnings)))
    errors=log.split('ERROR:')
    if len(errors) == 1:
        if len(log)==len(errors[0]):
            errors=[]
    print('{} errors'.format(len(errors)))
    return errors

       
def send_mail_through_oracle(sent_to,mail_body,mail_subject):
#    mr=MailReceiver('Aa123456','C:\\Anaconda3\\Lib\\site-packages\\sber_pack')
#    mr.sendMessage(list(set(sent_to+['iskra_cvm@mail.ca.sbrf.ru'])), mail_subject,mail_body)
    mail_to=';'.join(sent_to)
    con=cx_Oracle.connect(get_iskra_con_str())
    cur=con.cursor()
    send="insert into atb_emails (mail_user, mail_to, mail_to_copy, mail_body, mail_subject, mail_status) values ('single_base_sales_kb','iskra_cvm@mail.ca.sbrf.ru','{}','{}','{}',1)".format(mail_to,mail_body,mail_subject)
    cur.execute(send)
    con.commit()
    cur.close()
    con.close()
       
def create_temp_table(temp_table,ora_table,connection):
    con=cx_Oracle.connect(connection)
    cur=con.cursor()
    try:
        cur.execute('create table {} as (select * from {} where 1=0)'.format(temp_table,ora_table))
    except cx_Oracle.DatabaseError as e:
        if 'ORA-00955' in str(e):
            drop_temp_table(temp_table,connection)
            create_temp_table(temp_table,ora_table,connection)
        else:
            raise e
    cur.close()
    con.close()

def drop_temp_table(temp_table,connection):
    con=cx_Oracle.connect(connection)
    cur=con.cursor()
    cur.execute('drop table {}'.format(temp_table))
    cur.close()
    con.close()
        
def table_coping(ora_table,temp_table,connection):
    con=cx_Oracle.connect(connection)
    cur=con.cursor()
    cur.execute('truncate table '+ora_table)
    cur.execute("begin execute immediate 'alter session enable parallel dml'; execute immediate 'alter session enable parallel ddl'; end;")
    cur.execute('insert /*+append parallel(a 16) full(a) no_index(a)*/ into {} a select /*+parallel(b 16)*/* from {} b'.format(ora_table,temp_table))
    con.commit()
    cur.execute("begin execute immediate 'alter session disable parallel dml'; execute immediate 'alter session disable parallel ddl'; end;")
    cur.execute('drop table '+temp_table)
    cur.close()
    con.close()  

def count_table(ora_table,connection):
    con=cx_Oracle.connect(connection)
    cur=con.cursor()
    cnt=cur.execute('select count(*) from '+ora_table).fetchone()[0]
    cur.close()
    con.close()
    return cnt

def check_n_rows(ora_table,temp_table,connection):
    con=cx_Oracle.connect(connection)
    cur=con.cursor()
    main_table_cnt=cur.execute('select count(*) from '+ora_table).fetchone()[0]
    temp_table_cnt=cur.execute('select count(*) from '+temp_table).fetchone()[0]
    nerr=round(max(main_table_cnt,temp_table_cnt)*5/100)
    cur.close()
    con.close()      
    return main_table_cnt - temp_table_cnt > nerr

def run_eg_project_with_rows_check(type_oper_id,egproject,ora_table,temp_table,job_num,log,connection,work_dir,delayto=None,send_to=[]):
    sasldr_log=work_dir+'\\sasldr_log.txt'
    start=datetime.datetime.now()
    error=''
    print(ora_table.upper())
    sasldr_logging(sasldr_log,'BEGIN')
    l_opid=StartOper(connection,type_oper_id)
    l_stid=RegPhase(connection,l_opid,'Проверка наличия EG проекта')
    try:
        if not exists(egproject):
            raise IOError('Project file %s DOES NOT EXISTS!' % egproject)
    except Exception as e:
        sasldr_logging(sasldr_log,'ERROR. Project loading: '+str(e))
        AddLog(connection, l_stid, 'E', 'Project loading: '+str(e))
        error='EG project not found'
    else:
        EndPhase(connection, l_stid)
        l_stid=RegPhase(connection,l_opid,'Создание временной таблицы')
        try:
            create_temp_table(temp_table,ora_table,connection)
        except Exception as e:
            sasldr_logging(sasldr_log,'ERROR. Temp table creating: '+str(e))
            AddLog(connection, l_stid, 'E', 'Temp table creating: '+str(e))
            error='Temp table not created'
        else:
            EndPhase(connection, l_stid)
            l_stid=RegPhase(connection,l_opid,'Выполнение проекта SAS EG')
            sasldr_logging(sasldr_log,'Temp table created')
            print('Executing...')
            try:
                app = wcc.DispatchEx("SASEGObjectModel.Application.7.1")
                prjObject = app.Open(egproject,"")
                prjObject.run
                prjObject.Save
                prjObject.Close
            except Exception as e:
                sasldr_logging(sasldr_log,'ERROR. Project executing: '+str(e))
                AddLog(connection, l_stid, 'E', 'Project executing: '+str(e))
                error='EG poject not executed'
            else:
                EndPhase(connection, l_stid)
                l_stid=RegPhase(connection,l_opid,'Проверка выполнения проекта')
                try:
                    errors=log_parser(log)
                    new_name=log[:-4]+'({}).txt'.format(start.strftime('%d.%m.%Y'))
                    if exists(new_name):
                        os.unlink(new_name)
                    try:
                        os.rename(log,new_name)
                    except Exception as e:
                        sasldr_logging(sasldr_log,'ERROR WHEN RENAME LOG: '+str(e))
                except Exception as e:
                    sasldr_logging(sasldr_log,'ERROR. Log reading: '+str(e))
                    AddLog(connection, l_stid, 'E', 'Log reading: '+str(e))
                    error='EG poject not executed'
                else:
                    if len(errors)!=0:
                        AddLog(connection, l_stid, 'E', 'SAS errors: '+str(errors))
                        sasldr_logging(sasldr_log,'SAS errors: ' + str(errors))
                        drop_temp_table(temp_table,connection)
                        error='SAS error (SAS loged on TB160000078555 {})'.format(new_name)
                    else:
                        EndPhase(connection, l_stid)
                        l_stid=RegPhase(connection,l_opid,'Проверка количества строк')
                        sasldr_logging(sasldr_log,'SAS EG project executed without any errors')
                        temp_cnt=count_table(temp_table,connection)
                        if check_n_rows(ora_table,temp_table,connection):
                            sasldr_logging(sasldr_log,'ERROR. New data much less then old')
                            error='New data much less then old'
                            AddLog(connection, l_stid, 'E', 'New data much less then old')
                        else:
                            try:
                                if delayto:
                                    EndPhaseWithRows(connection, l_stid, temp_cnt)
                                    l_stid=RegPhase(connection,l_opid,'Задержка')
                                    day=int(delayto.split(' ')[0])
                                    if day>6:
                                        day=6
                                    dl=delayto.split(' ')[1]
                                    d=datetime.date.today()
                                    while d.weekday() != day:
                                        d+=datetime.timedelta(1)
                                    delay=datetime.datetime.strptime(str(d)+' '+dl,'%Y-%m-%d %H:%M:%S')
                                    if delay>datetime.datetime.now():
                                        sasldr_logging(sasldr_log,'Delayed to: '+str(delay))
                                        delay=round((delay-datetime.datetime.now()).total_seconds())
                                        time.sleep(delay)
                                        print('Resumed')
                                else:
                                    EndPhaseWithRows(connection, l_stid, temp_cnt)
                            except Exception as e:
                                sasldr_logging(sasldr_log,'ERROR. Error with delay: '+str(e))
                                AddLog(connection, l_stid, 'E', 'Error with delay: '+str(e))
                                error='Error with delay'
                            else:
                                EndPhase(connection, l_stid)
                                l_stid=RegPhase(connection,l_opid,'Копирование данных в боевую таблицу')
                                try:
                                    table_coping(ora_table,temp_table,connection)
                                except Exception as e:
                                    sasldr_logging(sasldr_log,'ERROR. Data not copied to main table: '+str(e))
                                    AddLog(connection, l_stid, 'E', 'Data not copied to main table: '+str(e))
                                    error='Data not copied to main table'
                                else:
                                    ora_cnt=count_table(ora_table,connection)
                                    EndPhaseWithRows(connection, l_stid, ora_cnt)
                                    EndOper(connection, l_opid)
    try:
        oracle_logging(job_num,start,error)
    except Exception as e:
        sasldr_logging(sasldr_log,'ERROR. Writing logs: '+str(e))
    try:
        if error != '':
            send_mail_through_oracle(send_to,'Данные не загружены в {} Ошибка: {}'.format(ora_table,error),'SAS Loader')
        else:
            send_mail_through_oracle(send_to,'Данные успешно загружены в '+ora_table,'SAS Loader')
    except Exception as e:
        sasldr_logging(sasldr_log,'ERROR. Sending logs: '+str(e))
    sasldr_logging(sasldr_log,'END')


def run_eg_project(type_oper_id, egproject, ora_table, temp_table, job_num, log, connection, work_dir, delayto=None, send_to=[]):
    sasldr_log=work_dir+'\\sasldr_log.txt'
    start=datetime.datetime.now()
    error=''
    sasldr_logging(sasldr_log,ora_table.upper())
    sasldr_logging(sasldr_log,'BEGIN')
    l_opid=StartOper(connection,type_oper_id)
    l_stid=RegPhase(connection,l_opid,'Проверка наличия EG проекта')
    try:
        if not exists(egproject):
            raise IOError('Project file %s DOES NOT EXISTS!' % egproject)
    except Exception as e:
        sasldr_logging(sasldr_log,'ERROR. Project loading: '+str(e))
        AddLog(connection, l_stid, 'E', 'Project loading: '+str(e))
        error='EG project not found'
    else:
        EndPhase(connection, l_stid)
        l_stid=RegPhase(connection,l_opid,'Создание временной таблицы')
        try:
            create_temp_table(temp_table,ora_table,connection)
        except Exception as e:
            sasldr_logging(sasldr_log,'ERROR. Temp table creating: '+str(e))
            AddLog(connection, l_stid, 'E', 'Temp table creating: '+str(e))
            error='Temp table not created'
        else:
            EndPhase(connection, l_stid)
            l_stid=RegPhase(connection,l_opid,'Выполнение проекта SAS EG')
            sasldr_logging(sasldr_log,'Temp table created')
            sasldr_logging(sasldr_log,'Executing...')
            try:
                app = wcc.DispatchEx("SASEGObjectModel.Application.7.1")
                prjObject = app.Open(egproject,"")
                prjObject.run
                prjObject.Save
                prjObject.Close
            except Exception as e:
                sasldr_logging(sasldr_log,'ERROR. Project executing: '+str(e))
                AddLog(connection, l_stid, 'E', 'Project executing: '+str(e))
                error='EG poject not executed'
            else:
                EndPhase(connection, l_stid)
                l_stid=RegPhase(connection,l_opid,'Проверка выполнения проекта')
                try:
                    errors=log_parser(log)
                    new_name=log[:-4]+'({}).txt'.format(start.strftime('%d.%m.%Y'))
                    if exists(new_name):
                        os.unlink(new_name)
                    try:
                        os.rename(log,new_name)
                    except Exception as e:
                        sasldr_logging(sasldr_log,'ERROR WHEN RENAME LOG: '+str(e))
                except Exception as e:
                    sasldr_logging(sasldr_log,'ERROR. Log reading: '+str(e))
                    AddLog(connection, l_stid, 'E', 'Log reading: '+str(e))
                    error='EG poject not executed'
                else:
                    if len(errors)!=0:
                        AddLog(connection, l_stid, 'E', 'SAS errors: '+str(errors))
                        sasldr_logging(sasldr_log,'SAS errors: ' + str(errors))
                        drop_temp_table(temp_table,connection)
                        error='SAS error (SAS loged on TB160000078555 {})'.format(new_name)
                    else:
                        EndPhase(connection, l_stid)
                        l_stid=RegPhase(connection,l_opid,'Задержка')
                        sasldr_logging(sasldr_log,'SAS EG project executed without any errors')
                        try:
                            if delayto:
                                EndPhase(connection, l_stid)
                                l_stid=RegPhase(connection,l_opid,'Задержка')
                                day=int(delayto.split(' ')[0])
                                if day>6:
                                    day=6
                                dl=delayto.split(' ')[1]
                                d=datetime.date.today()
                                while d.weekday() != day:
                                    d+=datetime.timedelta(1)
                                delay=datetime.datetime.strptime(str(d)+' '+dl,'%Y-%m-%d %H:%M:%S')
                                if delay>datetime.datetime.now():
                                    sasldr_logging(sasldr_log,'Delayed to: '+str(delay))
                                    delay=round((delay-datetime.datetime.now()).total_seconds())
                                    time.sleep(delay)
                                    sasldr_logging(sasldr_log,'Resumed')
                        except Exception as e:
                            sasldr_logging(sasldr_log,'ERROR. Error with delay: '+str(e))
                            AddLog(connection, l_stid, 'E', 'Error with delay: '+str(e))
                            error='Error with delay'
                        else:
                            EndPhase(connection, l_stid)
                            l_stid=RegPhase(connection,l_opid,'Копирование данных в боевую таблицу')
                            try:
                                table_coping(ora_table,temp_table,connection)
                            except Exception as e:
                                sasldr_logging(sasldr_log,'ERROR. Data not copied to main table: '+str(e))
                                AddLog(connection, l_stid, 'E', 'Data not copied to main table: '+str(e))
                                error='Data not copied to main table'
                            else:
                                EndPhase(connection, l_stid)
                                EndOper(connection, l_opid)
    try:
        oracle_logging(job_num,start,error)
    except Exception as e:
        sasldr_logging(sasldr_log,'ERROR. Writing logs: '+str(e))
    try:
        if error != '':
            send_mail_through_oracle(send_to,'Данные не загружены в {} Ошибка: {}'.format(ora_table,error),'SAS Loader')
        else:
            send_mail_through_oracle(send_to,'Данные успешно загружены в '+ora_table,'SAS Loader')
    except Exception as e:
        sasldr_logging(sasldr_log,'ERROR. Sending logs: '+str(e))
    sasldr_logging(sasldr_log,'END')

   
def execute_eg_project(type_oper_id, egproject, project_title, job_num, log_path, work_dir, send_to=[]):
    connection=get_iskra_con_str()
    sasldr_log=work_dir+'\\sasldr_log.txt'
    start=datetime.datetime.now()
    error=''
    sasldr_logging(sasldr_log,project_title.upper())
    sasldr_logging(sasldr_log,'BEGIN')
    l_opid=StartOper(connection,type_oper_id)
    l_stid=RegPhase(connection,l_opid,'Проверка наличия EG проекта')
    try:
        if not exists(egproject):
            raise IOError('Project file %s DOES NOT EXISTS!' % egproject)
    except Exception as e:
        sasldr_logging(sasldr_log,'Project loading: '+str(e))
        AddLog(connection, l_stid, 'E', 'Project loading: '+str(e))
        error='EG project not found'
    else:
        EndPhase(connection, l_stid)
        l_stid=RegPhase(connection,l_opid,'Выполнение проекта SAS EG')
        sasldr_logging(sasldr_log,'SAS EG project found')
        sasldr_logging(sasldr_log,'Executing...')
        try:
            app = wcc.DispatchEx("SASEGObjectModel.Application.7.1")
            prjObject = app.Open(egproject,"")
            prjObject.run
            prjObject.Save
            prjObject.Close
        except Exception as e:
            sasldr_logging(sasldr_log,'Project executing: '+str(e))
            AddLog(connection, l_stid, 'E', 'Project executing: '+str(e))
            error='EG poject not executed'
        else:
            EndPhase(connection, l_stid)
            l_stid=RegPhase(connection,l_opid,'Проверка выполнения проекта')
            try:
                errors=log_parser(log_path)
                new_name=log_path[:-4]+'({}).txt'.format(start.strftime('%d.%m.%Y'))
                if exists(new_name):
                    os.unlink(new_name)
                try:
                    os.rename(log_path,new_name)
                except Exception as e:
                    sasldr_logging(sasldr_log,'ERROR WHEN RENAME LOG: '+str(e))
            except Exception as e:
                sasldr_logging(sasldr_log,'ERROR. Log reading: '+str(e))
                AddLog(connection, l_stid, 'E', 'Log reading: '+str(e))
                error='EG poject not executed'
            else:
                if len(errors)!=0:
                    sasldr_logging(sasldr_log,'SAS errors: ' + str(errors))
                    AddLog(connection, l_stid, 'E', 'SAS errors: ' + str(errors))
                    error='SAS error (SAS loged on TB160000078555 {})'.format(new_name)
                else:
                    sasldr_logging(sasldr_log,'SAS EG project executed without any errors')
                    EndPhase(connection, l_stid)
                    EndOper(connection, l_opid)
    try:
        oracle_logging(job_num,start,error)
    except Exception as e:
        sasldr_logging(sasldr_log,'ERROR. Writing logs: '+str(e))
    try:
        if error != '':
            send_mail_through_oracle(send_to,'Данные не загружены в {} Ошибка: {}'.format(project_title,error),'SAS Loader')
        else:
            send_mail_through_oracle(send_to,'Данные успешно загружены в '+project_title,'SAS Loader')
    except Exception as e:
        sasldr_logging(sasldr_log,'ERROR. Sending logs: '+str(e))
    sasldr_logging(sasldr_log,'END')

#def run_eg_project(egproject,ora_table,temp_table,job_num,log,connection,work_dir,delayto=None,send_to=[]):
#    sasldr_log=work_dir+'\\sasldr_log.txt'
#    start=datetime.datetime.now()
#    error=''
#    print(ora_table.upper())
#    sasldr_logging(sasldr_log,'BEGIN')
#    try:
#        if not exists(egproject):
#            raise IOError('Project file %s DOES NOT EXISTS!' % egproject)
#    except Exception as e:
#        sasldr_logging(sasldr_log,'ERROR. Project loading: '+str(e))
#        error='EG project not found'
#    else:
#        try:
#            create_temp_table(temp_table,ora_table,connection)
#        except Exception as e:
#            sasldr_logging(sasldr_log,'ERROR. Temp table creating: '+str(e))
#            error='Temp table not created'
#        else:
#            sasldr_logging(sasldr_log,'Temp table created')
#            print('Executing...')
#            try:
#                app = wcc.DispatchEx("SASEGObjectModel.Application.7.1")
#                prjObject = app.Open(egproject,"")
#                prjObject.run
#                prjObject.Save
#                prjObject.Close
#            except Exception as e:
#                sasldr_logging(sasldr_log,'ERROR. Project executing: '+str(e))
#                error='EG poject not executed'
#            else:
#                try:
#                    errors=log_parser(log)
#                    new_name=log[:-4]+'({}).txt'.format(start.strftime('%d.%m.%Y'))
#                    if exists(new_name):
#                        os.unlink(new_name)
#                    try:
#                        os.rename(log,new_name)
#                    except Exception as e:
#                        sasldr_logging(sasldr_log,'ERROR WHEN RENAME LOG: '+str(e))
#                except Exception as e:
#                    sasldr_logging(sasldr_log,'ERROR. Log reading: '+str(e))
#                    error='EG poject not executed'
#                else:
#                    if len(errors)!=0:
#                        
#                        sasldr_logging(sasldr_log,'SAS errors: ' + str(errors))
#                        drop_temp_table(temp_table,connection)
#                        error='SAS error (SAS loged on TB160000078555 {})'.format(new_name)
#                    else:
#                        sasldr_logging(sasldr_log,'SAS EG project executed without any errors')
#                        try:
#                            if delayto:
#                                day=int(delayto.split(' ')[0])
#                                if day>6:
#                                    day=6
#                                dl=delayto.split(' ')[1]
#                                d=datetime.date.today()
#                                while d.weekday() != day:
#                                    d+=datetime.timedelta(1)
#                                delay=datetime.datetime.strptime(str(d)+' '+dl,'%Y-%m-%d %H:%M:%S')
#                                if delay>datetime.datetime.now():
#                                    sasldr_logging(sasldr_log,'Delayed to: '+str(delay))
#                                    delay=round((delay-datetime.datetime.now()).total_seconds())
#                                    time.sleep(delay)
#                                    print('Resumed')
#                        except Exception as e:
#                            sasldr_logging(sasldr_log,'ERROR. Error with delay: '+str(e))
#                            error='Error with delay'
#                        else:
#                            try:
#                                table_coping(ora_table,temp_table,connection)
#                            except Exception as e:
#                                sasldr_logging(sasldr_log,'ERROR. Data not copied to main table: '+str(e))
#                                error='Data not copied to main table'
#    try:
#        oracle_logging(job_num,start,error)
#    except Exception as e:
#        sasldr_logging(sasldr_log,'ERROR. Writing logs: '+str(e))
#    try:
#        if error != '':
#            send_mail_through_oracle(send_to,'Данные не загружены в {} Ошибка: {}'.format(ora_table,error),'SAS Loader')
#        else:
#            send_mail_through_oracle(send_to,'Данные успешно загружены в '+ora_table,'SAS Loader')
#    except Exception as e:
#        sasldr_logging(sasldr_log,'ERROR. Sending logs: '+str(e))
#    sasldr_logging(sasldr_log,'END')

   
#def execute_eg_project(egproject,project_title,job_num,log_path,work_dir,send_to=[]):
#    sasldr_log=work_dir+'\\sasldr_log.txt'
#    start=datetime.datetime.now()
#    error=''
#    print(project_title.upper())
#    sasldr_logging(sasldr_log,'BEGIN')
#    try:
#        if not exists(egproject):
#            raise IOError('Project file %s DOES NOT EXISTS!' % egproject)
#    except Exception as e:
#        sasldr_logging(sasldr_log,'Project loading: '+str(e))
#        error='EG project not found'
#    else:
#        sasldr_logging(sasldr_log,'SAS EG project found')
#        print('Executing...')
#        try:
#            app = wcc.DispatchEx("SASEGObjectModel.Application.7.1")
#            prjObject = app.Open(egproject,"")
#            prjObject.run
#            prjObject.Save
#            prjObject.Close
#        except Exception as e:
#            sasldr_logging(sasldr_log,'Project executing: '+str(e))
#            error='EG poject not executed'
#        else:
#            try:
#                errors=log_parser(log_path)
#                new_name=log_path[:-4]+'({}).txt'.format(start.strftime('%d.%m.%Y'))
#                if exists(new_name):
#                    os.unlink(new_name)
#                try:
#                    os.rename(log_path,new_name)
#                except Exception as e:
#                    sasldr_logging(sasldr_log,'ERROR WHEN RENAME LOG: '+str(e))
#            except Exception as e:
#                sasldr_logging(sasldr_log,'ERROR. Log reading: '+str(e))
#                error='EG poject not executed'
#            else:
#                if len(errors)!=0:
#                    sasldr_logging(sasldr_log,'SAS errors: ' + str(errors))
#                    error='SAS error (SAS loged on TB160000078555 {})'.format(new_name)
#                else:
#                    sasldr_logging(sasldr_log,'SAS EG project executed without any errors')
#    try:
#        oracle_logging(job_num,start,error)
#    except Exception as e:
#        sasldr_logging(sasldr_log,'ERROR. Writing logs: '+str(e))
#    try:
#        if error != '':
#            send_mail_through_oracle(send_to,'Данные не загружены в {} Ошибка: {}'.format(project_title,error),'SAS Loader')
#        else:
#            send_mail_through_oracle(send_to,'Данные успешно загружены в '+project_title,'SAS Loader')
#    except Exception as e:
#        sasldr_logging(sasldr_log,'ERROR. Sending logs: '+str(e))
#    sasldr_logging(sasldr_log,'END')
    
    
def cvm_logging_start(kp_name, kp_log_name):
    con=cx_Oracle.connect(get_iskra_con_str())
    cur=con.cursor()
    res=cur.execute("select max(id_log) from cvm_log where project = 'SAS_EG_KP' and process_name='{}' and status=0".format(kp_log_name)).fetchone()[0]
    if res:
        l_log_id=res
    else:
        l_log_id=cur.callfunc('cvm_logging.log_start',int,[kp_log_name, kp_name, 'SAS_EG_KP'])
    cur.close()
    con.close()
    return l_log_id 
    
def cvm_logging_end(l_log_id, kp_log_name):
    con=cx_Oracle.connect(get_iskra_con_str())
    cur=con.cursor()
    cur.callproc('cvm_logging.log_end', [l_log_id])
    cur.callproc('cvm_check_p.start_checks_name',['SAS_EG_KP_00_MAIN_CHECKS', kp_log_name, False, False])
    cur.close()
    con.close()    

def cvm_logging_error(l_log_id, err_name, err_desc):
    con=cx_Oracle.connect(get_iskra_con_str())
    cur=con.cursor()
    cur.callproc('cvm_logging.log_error',[l_log_id, err_name, err_desc])
    cur.close()
    con.close() 

def cvm_logging_get_desc(kp_log_name):
    con=cx_Oracle.connect(get_iskra_con_str())
    cur=con.cursor()
    desc=cur.execute("select kp_name, kp_description from cvm_sas_eg_kp where kp_log_name = upper('{}')".format(kp_log_name)).fetchone()
    cur.close()
    con.close() 
    return desc

def cvm_logging_is_enable(kp_log_name):
    con=cx_Oracle.connect(get_iskra_con_str())
    cur=con.cursor()
    in_calc=cur.execute("select in_calc from cvm_sas_eg_kp where kp_log_name = upper('{}')".format(kp_log_name)).fetchone()[0]
    cur.close()
    con.close() 
    return bool(in_calc)

def cvm_logging_check(kp_log_name):
    con=cx_Oracle.connect(get_iskra_con_str())
    cur=con.cursor()
    check=cur.execute("""select max(log_status) keep (dense_rank last order by log_time)
                          from cvm_check_log_detail
                        where check_id = 341
                           and step_id = (select step_id
                                            from cvm_check_step
                                           where step_name = '{}')
                        """.format(kp_log_name)).fetchone()[0]
    cur.close()
    con.close()
    return check == 'OK'
   
def update_time(kp_log_name, taskname, sasldr_log):
    con=cx_Oracle.connect(get_iskra_con_str())
    cur=con.cursor()
    date_start=cur.execute(f"select date_start+2/24 from cvm_sas_eg_kp where kp_log_name = '{kp_log_name}'").fetchone()[0]
    cur.close()
    con.close()
    task=get_task_xml(taskname)
    task=re.sub('<StartBoundary>(.*)</StartBoundary>',f"<StartBoundary>{date_start.strftime('%Y-%m-%dT%H:%M:%S')}</StartBoundary>",task)
    sasldr_logging(sasldr_log,task)
    sasldr_logging(sasldr_log,delete_task(taskname))
    sasldr_logging(sasldr_log,task_from_xml(taskname, task))
#    sasldr_logging(sasldr_log,change_task(taskname, start_date=date_start.strftime('%d/%m/%Y'), start_time=date_start.strftime('%H:%M'), user='ALPHA\iskra_cvm', password='Aa123456'))
    
#def execute_cvm_sas_eg_kp(egproject,taskname,kp_log_name,job_num,log_path,work_dir,send_to=[]):
#    sasldr_log=work_dir+'\\sasldr_log.txt'
#    attempt=3
#    while attempt > 0:
#        attempt-=1
#        start=datetime.datetime.now()
#        error=''
#        print(kp_log_name.upper())
#        sasldr_logging(sasldr_log,'BEGIN')
#        kp_name,kp_desc=cvm_logging_get_desc(kp_log_name)
#        l_log_id=cvm_logging_start(kp_name,kp_log_name)
#        try:
#            if not cvm_logging_is_enable(kp_log_name):
#                raise Exception('DISABLED')
#        except Exception as e:
#            cvm_logging_error(l_log_id,'DISABLED',str(e))
#            sasldr_logging(sasldr_log,'DISABLED: '+str(e))
#            error='DISABLED'
#        else:        
#            try:
#                if not exists(egproject):
#                    raise IOError('Project file %s DOES NOT EXISTS!' % egproject)
#            except Exception as e:
#                cvm_logging_error(l_log_id,'Project loading',str(e))
#                sasldr_logging(sasldr_log,'Project loading: '+str(e))
#                error='EG project not found'
#            else:
#                sasldr_logging(sasldr_log,'SAS EG project found')
#                print('Executing...')
#                try:
#                    app = wcc.DispatchEx("SASEGObjectModel.Application.7.1")
#                    prjObject = app.Open(egproject,"")
#                    prjObject.run
#                    prjObject.Save
#                    prjObject.Close
#                except Exception as e:
#                    cvm_logging_error(l_log_id,'Project executing',str(e))
#                    sasldr_logging(sasldr_log,'Project executing: '+str(e))
#                    error='EG poject not executed'
#        try:
#            oracle_logging(job_num,start,error)
#        except Exception as e:
#            cvm_logging_error(l_log_id,'ERROR. Writing logs',str(e))
#            sasldr_logging(sasldr_log,'ERROR. Writing logs: '+str(e))
#        try:
#            if error != '':
#                send_mail_through_oracle(send_to,'Данные не загружены в {} Ошибка: {}'.format(kp_log_name,error),'SAS Loader')
#                cvm_logging_end(l_log_id,kp_log_name)
#                raise Exception('Данные не загружены')
#            else:
#                pass
##                send_mail_through_oracle(send_to,'Данные успешно загружены в '+kp_log_name,'SAS Loader')
#        except Exception as e:
#            sasldr_logging(sasldr_log,'ERROR. Sending logs: '+str(e))
#        else:
#            cvm_logging_end(l_log_id,kp_log_name)
#            try:
#                if cvm_logging_check(kp_log_name):
#                    send_mail_through_oracle(send_to,'Данные успешно загружены в {}. Данные корректны'.format(kp_log_name),'SAS Loader')
#                    break
#                else:
#                    send_mail_through_oracle(send_to,'Данные успешно загружены в {}, но при проверке найдена ошибка'.format(kp_log_name),'SAS Loader')
#            except Exception as e:
#                send_mail_through_oracle(send_to,'Данные успешно загружены в {}, но при проверке возникла ошибка'.format(kp_log_name),'SAS Loader')
#        sasldr_logging(sasldr_log,'END')
#    try:
#        sasldr_logging(sasldr_log,'RESET TIME')
#        update_time(kp_log_name, taskname, sasldr_log)
#    except Exception as e:
#        sasldr_logging(sasldr_log,'ERROR. Reseting time: '+str(e))

def execute_cvm_sas_eg_kp(type_oper_id, egproject, taskname, kp_log_name, job_num, log_path, work_dir, send_to=[]):
    connection=get_iskra_con_str()
    sasldr_log=work_dir+'\\sasldr_log.txt'
    attempt=3
    while attempt > 0:
        attempt-=1
        start=datetime.datetime.now()
        error=''
        sasldr_logging(sasldr_log,kp_log_name.upper())
        sasldr_logging(sasldr_log,'BEGIN')
        l_opid=StartOper(connection,type_oper_id)
        l_stid=RegPhase(connection,l_opid,'Проверка статуса SAS_EG_KP проекта')    
        kp_name,kp_desc=cvm_logging_get_desc(kp_log_name)
        l_log_id=cvm_logging_start(kp_name,kp_log_name)
        try:
            if not cvm_logging_is_enable(kp_log_name):
                raise Exception('DISABLED')
        except Exception as e:
            cvm_logging_error(l_log_id,'DISABLED',str(e))
            sasldr_logging(sasldr_log,'DISABLED: '+str(e))
            AddLog(connection, l_stid, 'E', 'DISABLED: '+str(e))
            error='DISABLED'
        else:
            EndPhase(connection, l_stid)
            l_stid=RegPhase(connection,l_opid,'Проверка наличия EG проекта')
            try:
                if not exists(egproject):
                    raise IOError('Project file %s DOES NOT EXISTS!' % egproject)
            except Exception as e:
                cvm_logging_error(l_log_id,'Project loading',str(e))
                AddLog(connection, l_stid, 'E', 'Project loading: '+str(e))
                sasldr_logging(sasldr_log,'Project loading: '+str(e))
                error='EG project not found'
            else:
                EndPhase(connection, l_stid)
                l_stid=RegPhase(connection,l_opid,'Выполнение проекта SAS EG')
                sasldr_logging(sasldr_log,'SAS EG project found')
                sasldr_logging(sasldr_log,'Executing...')
                try:
                    app = wcc.DispatchEx("SASEGObjectModel.Application.7.1")
                    prjObject = app.Open(egproject,"")
                    prjObject.run
                    prjObject.Save
                    prjObject.Close
                except Exception as e:
                    cvm_logging_error(l_log_id,'Project executing',str(e))
                    AddLog(connection, l_stid, 'E', 'Project executing: '+str(e))
                    sasldr_logging(sasldr_log,'Project executing: '+str(e))
                    error='EG poject not executed'
                else:
                    EndPhase(connection, l_stid)
                    EndOper(connection, l_opid)
        try:
            oracle_logging(job_num,start,error)
        except Exception as e:
            cvm_logging_error(l_log_id,'ERROR. Writing logs',str(e))
            sasldr_logging(sasldr_log,'ERROR. Writing logs: '+str(e))
        try:
            if error != '':
                send_mail_through_oracle(send_to,'Данные не загружены в {} Ошибка: {}'.format(kp_log_name,error),'SAS Loader')
                cvm_logging_end(l_log_id,kp_log_name)
                raise Exception('Данные не загружены')
            else:
                pass
#                send_mail_through_oracle(send_to,'Данные успешно загружены в '+kp_log_name,'SAS Loader')
        except Exception as e:
            sasldr_logging(sasldr_log,'ERROR. Sending logs: '+str(e))
        else:
            cvm_logging_end(l_log_id,kp_log_name)
            try:
                if cvm_logging_check(kp_log_name):
                    send_mail_through_oracle(send_to,'Данные успешно загружены в {}. Данные корректны'.format(kp_log_name),'SAS Loader')
                    break
                else:
                    send_mail_through_oracle(send_to,'Данные успешно загружены в {}, но при проверке найдена ошибка'.format(kp_log_name),'SAS Loader')
            except Exception as e:
                send_mail_through_oracle(send_to,'Данные успешно загружены в {}, но при проверке возникла ошибка'.format(kp_log_name),'SAS Loader')
        sasldr_logging(sasldr_log,'END')
    try:
        sasldr_logging(sasldr_log,'RESET TIME')
        update_time(kp_log_name, taskname, sasldr_log)
    except Exception as e:
        sasldr_logging(sasldr_log,'ERROR. Reseting time: '+str(e))