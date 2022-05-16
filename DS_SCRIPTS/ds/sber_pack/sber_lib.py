from datetime import datetime
import re
import pandas as pd
import cx_Oracle
from threading import Thread,Lock
import xlwt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from email.utils import formatdate
from os.path import basename
import subprocess
import time
import numpy as np
import requests
from pandas.api.types import is_numeric_dtype
import urllib

   
def get_con_creds(iskra):
    if iskra == 'iskra1_primary':
        return 'tech_iskra[iskra]','Uthvfy123','iskra1_primary'    
    if iskra == 'iskra2':
        return 'tech_iskra[iskra]','Uthvfy123','iskra2'    
    if iskra == 'not13_igrand':
        return 'tech_iskra[iskra]','Uthvfy123','not13_igrand'        
    if iskra == 'iskra4':
        return 'tech_iskra[iskra]','Uthvfy123','iskra4'

def get_iskra_con_str(iskra='iskra4'):
    return '{}/{}@{}'.format(*get_con_creds(iskra))
    
def sber_service_auth(service, domain, username, password):
    requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
       
    head={
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Encoding':'gzip, deflate, br',
    'Accept-Language':'ru,en;q=0.8',
    'Cache-Control':'max-age=0',
    'Connection':'keep-alive',
    'Content-Length':'96',
    'Content-Type':'application/x-www-form-urlencoded',
    'Host':'motivation-kb.ca.sbrf.ru',
    'Origin':'https://'+service,
    'Referer':'https://'+service,
    'Upgrade-Insecure-Requests':'1',
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.125 YaBrowser/17.7.1.791 Yowser/2.5 Safari/537.36'}
    
    data='domain={}&username={}&password={}&login-form-type=pwd'.format(domain, username, urllib.parse.quote(password))
    
    session=requests.Session()
    
    res=session.post('https://{}/pkmslogin.form'.format(service), headers=head, data=data, verify=False)
    print(res)
    print(res.text)
    return session

def generate_table_script(table_name,dframe):
    cols=[]
    for c in dframe:
        new_c=re.sub('\W','_',c).lower()
        if len(new_c) > 30:
            for a in ['a','e','y','u','i','o']:
                new_c=new_c.replace(a,'')
        if is_numeric_dtype(dframe[c]):
            cols.append('{} number'.format(new_c))
        else:
            max_len=dframe[c].fillna('').replace({'nan':''}).apply(len).max()
            if max_len > 4000:
                cols.append('{} clob'.format(new_c))
            elif max_len == 0:
                cols.append('{} varchar2({})'.format(new_c, 30))
            else:
                cols.append('{} varchar2({})'.format(new_c, int(max_len+max_len*0.3 if max_len+max_len*0.3 < 4000 else 4000)))
    return 'create table {} ({})'.format(table_name,', '.join(cols))

def trans_generate_table_script(table_name,dframe):
    cols=[]
    for c in dframe:
        new_c=re.sub('\W','_',translate(c)).lower()
        new_c=re.sub('_+','_',new_c)
        if len(new_c) > 30:
            for a in ['a','e','y','u','i','o']:
                new_c=new_c.replace(a,'')
        while len(new_c) > 30:
            new_c='_'.join(new_c.split('_')[:-1])
        if is_numeric_dtype(dframe[c]):
            cols.append('{} number'.format(new_c))
        else:
            max_len=dframe[c].fillna('').replace({'nan':''}).apply(len).max()
            if max_len > 4000:
                cols.append('{} clob'.format(new_c))
            elif max_len == 0:
                cols.append('{} varchar2({})'.format(new_c, 30))
            else:
                cols.append('{} varchar2({})'.format(new_c, int(max_len+max_len*0.3 if max_len+max_len*0.3 < 4000 else 4000)))
    return 'create table {} ({})'.format(table_name,', '.join(cols))

def oracle_logging(job_num,start,error=''):
    con=cx_Oracle.connect(get_iskra_con_str())
    cur=con.cursor()
    
    status='COMPLETE' if error == '' else 'ERROR'
    job_broken='N' if status == 'COMPLETE' else 'Y'
    load_info_upd="update atb_segmen_load_info set t_last_load=sysdate, t_next_load=load_info_next_date(t_interval), t_job_broken='{}' where t_job={}".format(job_broken,job_num)

    duration=round((datetime.now()-start).total_seconds())
    ses=cur.execute("select SYS_CONTEXT('USERENV', 'SESSIONID') from dual")
    session_id=ses.fetchone()[0]
    job_run_details_ins="insert into job_run_details (log_id,log_date,owner, job, status, actual_start_date, actual_end_date, run_duration, instance_id, session_id, additional_info) values (job_run_details_seq.NEXTVAL, sysdate, 'ISKRA', '{}', '{}', to_date('{}','dd.mm.yyyy HH24:MI:SS'), to_date('{}','dd.mm.yyyy HH24:MI:SS'), numtodsinterval({},'second'), 1,'{}','{}')".format(job_num,status,start.strftime('%d.%m.%Y %H:%M:%S'),datetime.now().strftime('%d.%m.%Y %H:%M:%S'),duration,session_id,error)

    cur.execute(load_info_upd)
    con.commit()

    cur.execute(job_run_details_ins)
    con.commit()
    
    cur.close()
    con.close()

def open_imp_log(process_name, comment_text):
    try:
        con=cx_Oracle.connect(get_iskra_con_str())
        cur=con.cursor()
        res=cur.execute("select cvm_import_eksmb_logging.log_start_process('{}', '{}', null, null) from dual".format(process_name.lower(), comment_text))
        imp_log_id=res.fetchone()[0]
        cur.close()
        con.close()
        return imp_log_id
    except:
        return 000000000000000
    
def update_imp_log(imp_log_id, comment_text, processing):
    try:
        con=cx_Oracle.connect(get_iskra_con_str())
        cur=con.cursor()
        cur.execute("begin cvm_import_eksmb_logging.log_update_process({}, '{}', {}); end;".format(imp_log_id, comment_text, processing))
        cur.close()
        con.close()
    except:
        pass
    
def close_imp_log(imp_log_id, comment_text):
    try:
        con=cx_Oracle.connect(get_iskra_con_str())
        cur=con.cursor()
        cur.execute("begin cvm_import_eksmb_logging.log_end_process({}, '{}'); end;".format(imp_log_id, comment_text))
        cur.close()
        con.close()
    except:
        pass
    
def close_error_imp_log(imp_log_id, comment_text, error):
    try:
        con=cx_Oracle.connect(get_iskra_con_str())
        cur=con.cursor()
        cur.execute("begin cvm_import_eksmb_logging.log_end_process_error({}, null, '{}'); end;".format(imp_log_id, error))
        cur.close()
        con.close()
    except:
        pass
    
def get_columns(table):
    con=cx_Oracle.connect(get_iskra_con_str())
    cols=pd.read_sql("select column_name from user_tab_columns where lower(table_name)='{}'".format(table.lower()),con=con).values.ravel()
    con.close()
    return ','.join(cols)

def parallel_sqlldr_with_ctl(connection, ctl_full_path):
    try:
        cmd = 'sqlldr.exe userid={} control="{}"'.format(connection,ctl_full_path)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
    except Exception as e:
       print(e)
    finally:
        print(out.decode('cp1251'))
        print(err.decode('cp1251'))

def parallel_sqlldr_without_ctl(connection, data, delimiter, table_name):
    try:
        if type(data)==pd.core.frame.DataFrame:
            data.to_csv('data_for_sqlldr.csv', delimiter=';',header=None, index=None)
            file='data_for_sqlldr.csv'
        elif type(data)==str:
            file=data
        else:
            raise Exception
        con=cx_Oracle.connect(connection)
        col=pd.read_sql("select column_name, data_type, data_length from user_tab_columns where lower(table_name) like '{}'".format(table_name), con=con)
        part1=col.loc[(col['DATA_TYPE']=='VARCHAR2')&(col['DATA_LENGTH']==4000)].copy()
        part2=col.loc[(col['DATA_TYPE']!='VARCHAR2')|(col['DATA_LENGTH']!=4000)].copy()
        part1['DATA_TYPE']=' CHAR(4000)'
        part2['DATA_TYPE']=''
        col=pd.concat([part1,part2])
        col=col.sort_index()
        cols=''
        for i in col.index:
            cols+='{}{}, '.format(col['COLUMN_NAME'][i],col['DATA_TYPE'][i])
        cols=cols[:len(cols)-2]
        ctl='LOAD DATA\nCHARACTERSET CL8MSWIN1251\nINFILE "{}"\nAPPEND\nINTO TABLE {}\nFIELDS TERMINATED BY \'{}\' OPTIONALLY ENCLOSED BY \'"\' TRAILING NULLCOLS\n({})\n\n'.format(file,table_name,delimiter,cols)
        with open('load.ctl','w') as w:
            w.write(ctl)
            w.close()
        try:
            cmd = 'sqlldr.exe userid={} control=load.ctl'.format(connection)
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = p.communicate()
        except Exception as e:
           print(e)
        finally:
            print(out.decode('cp1251'))
            print(err.decode('cp1251'))
    except Exception as e:
        print(e)

class SimpleList(list):
    def __init__(self, header, Column):
        super(SimpleList, self).__init__()
        self.header = header
        self.Column = Column

    def close(self):
        del self

class List(object):
    def __init__(self, num):
        self.num = num
        self.__chunks = []
        self.i = 0
        self.header = None
        self.Column = None

    def append(self, val):
        # if self.i == self.num: self.i=0
        i = self.i % self.num
        if len(self.__chunks) <= i:
            self.__chunks.append(SimpleList(self.header, self.Column))
        self.__chunks[i].append(val)
        self.i+=1

    def split(self): return self.__chunks

    @property
    def chunks(self): return self.__chunks

    # @property
    # def num(self): return len(self.__chunks)

    def __len__(self):
        return sum([len(c) for c in self.__chunks])

    def __iter__(self):
        for c in self.__chunks:
            for r in c:
                yield r

def conv(fld, tpe=None):
    if hasattr(fld,'value'):
        val = fld.value
    else:
        val = fld
    if (val == 'NULL') | (val==''):
        return None
    try:
        if tpe.lower() == 'number':
            return float(val.replace(',','.'))
        elif 'varchar' in tpe.lower():
            return val
        elif tpe.lower() == 'date':
            try:
                return datetime.strptime(val, "%d.%m.%Y")
            except:
                try:
                    return datetime.strptime(val, "%Y-%m-%d")
                except:
                    try:
                        return datetime.strptime(val, "%d.%m.%y")
                    except:
                        try:
                            return datetime.strptime(val, "%d.%m.%Y %H:%M:%S")
                        except:
                            try:
                                return datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
                            except:
                                try:
                                    return datetime.strptime(val, "%Y-%m-%dT%H:%M:%S")
                                except:
                                    try:
                                        return datetime.strptime(val, "%d.%m.%Y %H:%M")
                                    except:
                                        return ''
    except Exception as e:
            print(e)
            return nvl(val)
    return nvl(val)

def convrow(row, types, cols=None):
    if cols is None:
        return [conv(f,t) for f, t in zip(row, types)]
    else:
        outrow = []
        for c in cols:
            outrow.append(conv(row[c]))
        return outrow

def nvl(val):
    return val if not val is None else ""

def run(data, pool, ins, types):
    con = pool.acquire()
    cur = con.cursor()
    cur.prepare(ins)
    if True:
        for i, r in enumerate(data):
            try:
                cur.execute(None, convrow(r,types))
                if i % 1000 == 0:
                    print(i)
            except Exception as e:
                print(str(e))
        con.commit()


def parallel_uploading(user,password,schema,THREADS, data, table_name,columns_list=None):
#    data=data.astype(str)
#    data=data.replace({'nan':None}).fillna('')
    if user == '':
        user, password, schema = get_con_creds(schema)
    lst = List(THREADS)
    for line in data.values:
        lst.append(line)
    pool = cx_Oracle.SessionPool(user, password, schema, 1, THREADS if data.shape[0] > 1 else 2, 1, threaded=True)
    con = pool.acquire()
#    col=pd.read_sql('select * from {} where rownum<1'.format(table_name),con=con).columns
    col=pd.read_sql("select column_name, data_type from all_tab_columns where lower(table_name) like '%s' order by column_id" % table_name.lower(), con=con)
    pool.release(con)

    if columns_list:
        col=col.loc[col['COLUMN_NAME'].isin([c.upper() for c in columns_list])]
    
    ins = """insert into %(table_name)s (%(fields)s) values (%(datarow)s)""" % { 'table_name': table_name,
                                    'datarow': ", ".join(":"+str(i) for i in range(len(col['COLUMN_NAME'].values))),
                                    'fields': ', '.join(col['COLUMN_NAME'].values)}
    threads = []
    for i in range(THREADS):
        t = Thread(target = run, args=(lst.chunks[i], pool, ins, col['DATA_TYPE'].values))
        threads.append(t)
        
        t.start()
    for t in threads:
        t.join()
    del pool
    try:
        con=cx_Oracle.connect('{}/{}@{}'.format(user,password,schema))
#        cur=con.cursor()
#        cnt=cur.execute('select count(*) from {}'.format(table_name)).fetchone()[0]
#        cur.close()
        con.close()
#        if cnt != data.shape[0]:
#            raise Exception('Data uploading incomplete')
    except Exception as e:
#        print('error while checking results')
#        raise e
        pass

def to_excel(filename,df):
    book=xlwt.Workbook()
    sh=book.add_sheet('Sheet1')
    j=0
    for c in df.columns:
        i=0
        sh.write(i,j,c)
        for v in df[c]:
            i+=1
            sh.write(i,j,v)
        j+=1

    book.save(filename)

def send_mail(url,domain_and_user, pswd, send_from, send_to, text, subject, list_of_attachments=[]):
    try:
        msg=MIMEMultipart()
        msg['From']=send_from
        msg['To']=send_to
        msg['Date']=formatdate(localtime=True)
        msg['Subject']=subject
        msg.attach(MIMEText(text))
        for f in list_of_attachments:
            with open(f,'rb') as file:
                part=MIMEApplication(file.read(),Name=basename(f))
                part['Content-Disposition']='attachment; filename="%s"' % basename(f)
                msg.attach(part)
        smtp=smtplib.SMTP(url)
        smtp.ehlo()
        smtp.starttls()
        smtp.login(domain_and_user, pswd)
        smtp.sendmail(send_from,send_to,msg.as_string())
        smtp.close
        print('Sent')
    except Exception as e:
        print('Error:'+str(e))
        pass

def translate(s):
    letters=pd.DataFrame()
    letters['RU']=[u'а',u'б',u'в',u'г',u'д',u'е',u'ё',u'ж',u'з',u'и',u'й',u'к',u'л',u'м',u'н',u'о',u'п',u'р',u'с',u'т',u'у',u'ф',u'х',u'ц',u'ч',u'ш',u'щ',u'ъ',u'ы',u'ь',u'э',u'ю',u'я']
    letters['EN']=['a','b','v','g','d','e','e','zh','z','i','y','k','l','m','n','o','p','r','s','t','u','f','h','ts','ch','sh','shch','','y','','e','yu','ya']
    words=s.split()
    ts=''
    if re.search('[а-я]', s.lower()):
        for c in s.lower():
            if c in letters['RU'].values:
                ts+=letters[letters['RU']==c]['EN'].values[0]
            else:
                ts+=c
        return ' '.join(ts.split())
#        return s
    elif re.search('[a-z]', s.lower()):
        for w in words:
            if re.search('[а-я]', w):
                ts+=w
                continue
            else:
                ind=0
                while(ind < len(w)):
                    for n in range(4, 0, -1):
                        st=w[ind:ind+n]
                        if len(st) > 0:
                            if st in letters['EN'].values:
                                ts+=letters[letters['EN']==st]['RU'].values[0]
                                ind+=n
                                break
                            elif n==1:
                                ind+=1
                                break
    else:
        return s
    return ' '.join(ts.split())

class AsyncExecute(Thread):
    def __init__(self, cur,query):
        Thread.__init__(self)
        self.cur=cur
        self.query=query
    def run(self):
        self.cur.execute(self.query)
        res=self.cur.fetchall()
        self.cur.close()
        mutex.acquire()
        global df
        df=df.append(res,ignore_index=True)
        mutex.release()

def parallel_downloading(table, batch_size,n_threads,connection):
    try:
        con=cx_Oracle.connect(connection,threaded=True)
        cur=con.cursor()
        cur.execute('select count(*) from {}'.format(table))
        cnt=cur.fetchall()[0][0]
        cur.close()
        queries=[]
        limit=int(np.ceil(cnt/batch_size))
        limit=limit if limit > 0 else 1
        for l in range(limit):
            queries.append('select *from (select tmp.*,rownum rn from(select * from {})tmp where rownum<={})where rn>{}'.format(table,(l+1)*batch_size,l*batch_size))
        global df
        df=pd.DataFrame()
        global mutex
        mutex=Lock()
        batch=int(np.ceil(len(queries)/n_threads))
        batch=batch if batch > 0 else 1
        for tl in range(batch):
            th=[]
            for i, query in enumerate(queries[tl*n_threads:(tl+1)*n_threads]):
                cur=con.cursor()
                th.append(AsyncExecute(cur,query))
                th[i].start()
            for t in th:
                t.join()
        df.drop(df.columns[-1],axis=1)
    except Exception as e:
        if str(e)=='':
            time.sleep(10)
            parallel_downloading(table, batch_size,n_threads,connection)
    except Exception as e:
        print(e)
    finally:
        con.close()
    return df
