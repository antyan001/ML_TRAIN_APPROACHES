import cx_Oracle
from os.path import exists
import win32com.client as wcc
import datetime
import os
import subprocess
import time
import pytz
from sber_pack.mail_utils import MailReceiver

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

def is_updating(connection, name, field):
    con=cx_Oracle.connect(connection)
    cur=con.cursor()
    res=cur.execute("""select {} from td_loadings where name = '{}'""".format(field,name)).fetchone()
    cur.close()
    con.close()
    if res[0] in ['UPDATING', 'CRITICAL']:
        return True
    else:
        return False

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

def sasldr_logging(path,message):
    print(message)
    with open(path, 'a') as l:
        l.write('{} {}\n'.format(str(datetime.datetime.now()),message))
        l.close()
        
def main_error_logging(path,error):
    with open(path,'w') as e:
        e.write(error)
        e.close()
    raise BaseException(error)

def run_eg_project(egproject,ora_table,temp_table,job_num,interval,interval_info,log,connection,work_dir,delayto=None,send_to='iskra_cvm@mail.ca.sbrf.ru'):
    sasldr_log=work_dir+'\\sasldr_log.txt'
    error_path=work_dir+'\\main_error.txt'
    start=datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S')
    err=''
    try:
        try:
            if not exists(egproject):
                raise IOError('Project file %s DOES NOT EXISTS!' % egproject)
        except Exception as e:
            sasldr_logging(sasldr_log,'Project loading: '+str(e))
            main_error_logging(error_path, '1|EG project not found')
        else:
            sasldr_logging(sasldr_log,'SAS EG project found')
            if create_temp_table(temp_table,ora_table,connection):
                    main_error_logging(error_path, '2|Temp table not created')
            print('Executing...')
            try:
                app = wcc.DispatchEx("SASEGObjectModel.Application.7.1")
                prjObject = app.Open(egproject,"")
                prjObject.run
                prjObject.Save
                prjObject.Close
            except Exception as e:
                sasldr_logging(sasldr_log,'Project executing: '+str(e))
                main_error_logging(error_path, '3|EG poject not executed')
            else:
                errors=log_parser(log)
                new_name=log[:-4]+'({}).txt'.format(start.split(' ')[0])
                if exists(new_name):
                    os.unlink(new_name)
                os.rename(log,new_name)
                if len(errors)!=0:
                    sasldr_logging(sasldr_log,'SAS errors: ' + str(errors))
                    drop_temp_table(temp_table,connection)
                    main_error_logging(error_path, '4|SAS error (SAS loged on TB160000078555 {})'.format(new_name))
                else:
                    sasldr_logging(sasldr_log,'SAS EG project executed without any Serrors')
                    if delayto:
                        try:
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
                        except Exception as e:
                            sasldr_logging(sasldr_log,'Error with delay: '+str(e))
                    if table_coping(ora_table,temp_table,connection):
                       main_error_logging(error_path, '5|Data not copied to main table')
    finally:
        oracle_logging(ora_table,job_num,interval,interval_info,start,connection,error_path)
        if exists(error_path):
            with open(error_path,'r') as e:
                error=e.read()
                e.close()
                send_mail_through_oracle(send_to,'Данные не загружены в {} Ошибка: {}'.format(ora_table,error),'SAS Loader',connection)
            os.unlink(error_path)
        else:
           send_mail_through_oracle(send_to,'Данные успешно загружены в '+ora_table,'SAS Loader',connection)


def send_mail_through_oracle(mail_to,mail_body,mail_subject,connection):
    mr=MailReceiver('Aa123456','C:\\Anaconda3\\Lib\\site-packages\\sber_pack')
    mr.sendMessage(list(set(mail_to.split(';')+['iskra_cvm@mail.ca.sbrf.ru'])), mail_subject,mail_body)
#    con=cx_Oracle.connect(connection)
#    cur=con.cursor()
#    try:
#        send="insert into atb_emails (mail_user, mail_to, mail_to_copy, mail_body, mail_subject, mail_status) values ('single_base_sales_kb','{}','iskra_cvm@mail.ca.sbrf.ru','{}','{}',1)".format(mail_to,mail_body,mail_subject)
#        cur.execute(send)
#        con.commit()
#    except Exception as e:
#        raise IOError(e)
       
def oracle_logging_gy1(job_num,start,connection,error_path):
    err=''
    try:
        con=cx_Oracle.connect(connection)
        cur=con.cursor()
    except Exception as e:
        err=e
    try:
        if not exists(error_path):
            error=''
            status='COMPLETE'
            errn=''
        else:
            with open(error_path,'r') as e:
                error=e.read()
                e.close()
            status='ERROR'
            errn=error.split('|')[0]
            error=error.split('|')[1]
        last_date=datetime.datetime.now()
        duration=last_date-datetime.datetime.strptime(start, '%d.%m.%Y %H:%M:%S')
        duration=round(duration.total_seconds())
        last_date=last_date.strftime('%d.%m.%Y %H:%M:%S')
        ses=cur.execute("select SYS_CONTEXT('USERENV', 'SESSIONID') from dual")
        session_id=[s[0] for s in ses][0]
        cpu=cur.execute("SELECT ss.value FROM v$sesstat ss, v$statname sn, v$session s WHERE ss.statistic# = sn.statistic# and ss.sid = s.sid AND sn.name = 'CPU used by this session' AND s.audsid = SYS_CONTEXT('USERENV', 'SESSIONID')")
        cpu_used=[c[0] for c in cpu][0]
        seq=cur.execute('select job_run_details_seq.NEXTVAL from dual')
        seq_id=[s[0] for s in seq][0]
        job_run_details_ins="insert into job_run_details (log_id, log_date, owner, job, status, error#, actual_start_date,actual_end_date,run_duration,instance_id,session_id,cpu_used,additional_info) values ({},to_date('{}','dd.mm.yyyy HH24:MI:SS'),'{}','{}','{}','{}',to_date('{}','dd.mm.yyyy HH24:MI:SS'),to_date('{}','dd.mm.yyyy HH24:MI:SS'),numtodsinterval({},'second'),'{}','{}','{}','{}')".format(seq_id,last_date,'iskra',job_num,status,errn,start,last_date,duration,1,session_id,int(cpu_used),error)

        cur.execute(job_run_details_ins)
        con.commit()
    except Exception as e:
        raise IOError(e)
    if err!='':
        raise BaseException(err)

def oracle_logging(ora_table,job_num,interval,interval_info,start,connection,error_path):
    err=''
    try:
        con=cx_Oracle.connect(connection)
        cur=con.cursor()
    except Exception as e:
        err=e
    try:
        if not exists(error_path):
            error=''
            status='COMPLETE'
            errn=''
        else:
            with open(error_path,'r') as e:
                error=e.read()
                e.close()
            status='ERROR'
            errn=error.split('|')[0]
            error=error.split('|')[1]
        last_date=datetime.datetime.now()
        next_date=datetime.datetime.now()+datetime.timedelta(days=int(interval))
        next_date=next_date.strftime('%d.%m.%Y %H:%M:%S')
        duration=last_date-datetime.datetime.strptime(start, '%d.%m.%Y %H:%M:%S')
        duration=round(duration.total_seconds())
        last_date=last_date.strftime('%d.%m.%Y %H:%M:%S')
        load_info_upd="update atb_segmen_load_info set (t_last_load, t_next_load, t_job_broken)=(select to_date('{}','dd.mm.yyyy HH24:MI:SS'),to_date('{}','dd.mm.yyyy HH24:MI:SS'), 'N' from dual) where t_job={}".format(last_date,next_date,job_num)
        ses=cur.execute("select SYS_CONTEXT('USERENV', 'SESSIONID') from dual")
        session_id=[s[0] for s in ses][0]
        cpu=cur.execute("SELECT ss.value FROM v$sesstat ss, v$statname sn, v$session s WHERE ss.statistic# = sn.statistic# and ss.sid = s.sid AND sn.name = 'CPU used by this session' AND s.audsid = SYS_CONTEXT('USERENV', 'SESSIONID')")
        cpu_used=[c[0] for c in cpu][0]
        seq=cur.execute('select job_run_details_seq.NEXTVAL from dual')
        seq_id=[s[0] for s in seq][0]
        job_run_details_ins="insert into job_run_details (log_id, log_date, owner, job, status, error#, actual_start_date,actual_end_date,run_duration,instance_id,session_id,cpu_used,additional_info) values ({},to_date('{}','dd.mm.yyyy HH24:MI:SS'),'{}','{}','{}','{}',to_date('{}','dd.mm.yyyy HH24:MI:SS'),to_date('{}','dd.mm.yyyy HH24:MI:SS'),numtodsinterval({},'second'),'{}','{}','{}','{}')".format(seq_id,last_date,'iskra',job_num,status,errn,start, last_date,duration,1,session_id,int(cpu_used),error)

        cur.execute(load_info_upd)
        con.commit()

        cur.execute(job_run_details_ins)
        con.commit()
    except Exception as e:
        raise IOError(e)
    if err!='':
        raise BaseException(err)
    
def create_temp_table(temp_table,ora_table,connection):
    con=cx_Oracle.connect(connection)
    cur=con.cursor()
    try:
        cur.execute('create table {} as (select * from {} where 1=0)'.format(temp_table,ora_table))
    except Exception as e:
        print(e)
        raise IOError(e)

def drop_temp_table(temp_table,connection):
    con=cx_Oracle.connect(connection)
    cur=con.cursor()
    try:
            cur.execute('drop table {}'.format(temp_table))
    except Exception as e:
            print(e)
            raise IOError(e)
        
def table_coping(ora_table,temp_table,connection):
    try:
        con=cx_Oracle.connect(connection)
        cur=con.cursor()
        cur.execute('truncate table '+ora_table)
        cur.execute("begin execute immediate 'alter session enable parallel dml'; execute immediate 'alter session enable parallel ddl'; end;")
        cur.execute('insert /*+append parallel(a 16) full(a) no_index(a)*/ into {} a select /*+parallel(b 16)*/* from {} b'.format(ora_table,temp_table))
        con.commit()
        cur.execute("begin execute immediate 'alter session disable parallel dml'; execute immediate 'alter session disable parallel ddl'; end;")
    except Exception as e:
        print(e)
        err='Data not copied to main table'+str(e)+';'
        raise IOError(err)
    else:
        cur.execute('drop table '+temp_table)

def truncate_table(ora_table,connection):
    try:
        con=cx_Oracle.connect(connection)
        cur=con.cursor()
        cur.execute('truncate table '+ora_table)    
    except Exception as e:
        print(e)
        err='Table not truncated: '+str(e)+';'
        raise IOError(err)

def execute_statement(statement,connection):
    try:
        con=cx_Oracle.connect(connection)
        cur=con.cursor()
        cur.execute(statement)
    except Exception as e:
        err='Statement not executed: '+str(e)+';'
        raise IOError(err)

def execute_eg_project(work_dir,egproject,project_title,job_num,log_path,log_receiver=[]):
    send_to=';'.join(list(set(log_receiver+['iskra_cvm@mail.ca.sbrf.ru'])))
    sasldr_log=work_dir+'\\sasldr_log.txt'
    start=datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S')
    print(project_title)
    try:
        try:
            if not exists(egproject):
                raise IOError('Project file %s DOES NOT EXISTS!' % egproject)
        except Exception as e:
            sasldr_logging(sasldr_log,'Project loading: '+str(e))
            main_error_logging(error_path, '1|EG project not found')
        else:
            sasldr_logging(sasldr_log,'SAS EG project found')
            print('Executing...')
            try:
                app = wcc.DispatchEx("SASEGObjectModel.Application.7.1")
                prjObject = app.Open(egproject,"")
                prjObject.run
                prjObject.Save
                prjObject.Close
            except Exception as e:
                sasldr_logging(sasldr_log,'Project executing: '+str(e))
                main_error_logging(error_path, '2|EG poject not executed')
            else:
                errors=log_parser(log)
                new_name=log[:-4]+'({}).txt'.format(start.split(' ')[0])
                if exists(new_name):
                    os.unlink(new_name)
                os.rename(log,new_name)
                if len(errors)!=0:
                    sasldr_logging(sasldr_log,'SAS errors: ' + str(errors))
                    main_error_logging(error_path, '3|SAS error (SAS loged on TB160000078555 {})'.format(new_name))
                else:
                    sasldr_logging(sasldr_log,'SAS EG project executed without any errors')
    finally:
        oracle_logging(ora_table,job_num,start,error_path)
        if exists(error_path):
            with open(error_path,'r') as e:
                error=e.read()
                e.close()
                send_mail_through_oracle(send_to,'Данные не загружены в {} Ошибка: {}'.format(ora_table,error),'SAS Loader',connection)
            os.unlink(error_path)
        else:
           send_mail_through_oracle(send_to,'Данные успешно загружены в '+ora_table,'SAS Loader',connection)
