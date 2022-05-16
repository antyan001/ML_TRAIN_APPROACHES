import subprocess
import pandas as pd

def execute_cmd(cmd):
    try:
        p=subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode('866')
        return 'OUTPUT\n\n'+p
    except subprocess.CalledProcessError as e:
        return '!!!ERROR!!!\n\nstdin: {}\nstatuscode: {}\nstdout: {}'.format(e.cmd, e.returncode, e.stdout.decode('866'))

def create_task(name, trigger, executable, args=None, modificator=None, days=None, months=None, start_date=None, start_time=None, period=None, duration=None, end_date=None, end_time=None, user=None, password=None):
    cmd='schtasks /Create /TN {} /IT /SC {}'.format(name, trigger)
    if args:
        cmd+=' /TR "{} \\"{}\\""'.format(executable, args)
    else:
        cmd+=' /TR "{}"'.format(executable)        
    if modificator:
        cmd+=' /MO {}'.format(modificator)
    if days:
        cmd+=' /D {}'.format(days)
    if months:
        cmd+=' /M {}'.format(months)
    if start_date:
        cmd+=' /SD {}'.format(start_date)
    if start_time:
        cmd+=' /ST {}'.format(start_time)
    if period:
        cmd+=' /RI {}'.format(period)
    if duration:
        cmd+=' /DU {}'.format(duration)
    if end_date:
        cmd+=' /ED {}'.format(end_date)
    if end_time:
        cmd+=' /ET {}'.format(end_time)
    if user:
        cmd+=' /RU {}'.format(user)
    if password:
        cmd+=' /RP {}'.format(password)    
    
    return execute_cmd(cmd)

def change_task(name, enable=None, disable=None, executable=None, args=None, start_date=None, start_time=None, period=None, duration=None, end_date=None, end_time=None, user=None, password=None):
    cmd='schtasks /Change /TN {}'.format(name)
    if executable:
        if args:
            cmd+=' /TR "{} \\"{}\\""'.format(executable, args)
        else:
            cmd+=' /TR "{}"'.format(executable)        
    if start_date:
        cmd+=' /SD {}'.format(start_date)
    if start_time:
        cmd+=' /ST {}'.format(start_time)
    if period:
        cmd+=' /RI {}'.format(period)
    if duration:
        cmd+=' /DU {}'.format(duration)
    if end_date:
        cmd+=' /ED {}'.format(end_date)
    if end_time:
        cmd+=' /ET {}'.format(end_time)
    if enable:
        cmd+=' /Enable'
    if disable:
        cmd+=' /Disable'
    if user:
        cmd+=' /RU {}'.format(user)
    if password:
        cmd+=' /RP {}'.format(password)    
    return execute_cmd(cmd)

def run_task(name):
    return execute_cmd('schtasks /run /TN {}'.format(name))

def end_task(name):
    return execute_cmd('schtasks /end /TN {}'.format(name))

def delete_task(name):
    return execute_cmd('schtasks /delete /TN {} /F'.format(name))
    
def get_all_tasks(filters=None):
    p=subprocess.check_output('schtasks /query /fo csv /v', shell=True, stderr=subprocess.STDOUT).decode('866')    
    with open(r'C:\Temp\task_scheduler_table.csv', 'w',encoding='utf8') as file:
        file.write(p)
        file.close()
    df=pd.read_csv(r'C:\Temp\task_scheduler_table.csv',sep=',',encoding='utf8')
    df=df.drop_duplicates(keep=0)
    if filters:
        for k,v in filters.items():
            df=df.loc[df[k]==v]
    return df

def get_task_xml(taskname):
    return subprocess.check_output(f'schtasks /query /tn {taskname} /xml', shell=True, stderr=subprocess.STDOUT).decode('866')

def task_from_xml(taskname, xmltask):
    with open(r'C:\Temp\task.xml','w') as file:
        file.write(xmltask)
        file.close()
    res=execute_cmd(f'schtasks /create /tn {taskname} /xml C:\\Temp\\task.xml')
    if 'ОШИБКА' in res:
        raise Exception(res)
    return res
