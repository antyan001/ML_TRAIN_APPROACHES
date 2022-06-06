import os, binascii, random, string, shutil, sys
import keyring
from getpass import getpass
import zipfile
from time import sleep, strptime
from base64 import decodestring
from os.path import basename, dirname, join
from subprocess import call
from getpass import getpass
from datetime import date, datetime, timedelta
from glob import glob
import pandas as pd
import requests
from exchangelib import DELEGATE, Account, Credentials, Configuration, NTLM, Message, HTMLBody
from exchangelib.folders import FileAttachment, Mailbox, MeetingRequest #, Message
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import win32com
import win32com.client
import pywintypes as pwt
workdir = 'C:\\Temp'

processdir = "Proccessed"
errdir = "Errors"
out = "Out"
logdir = "Log"
notload = "notload"

def GetPass(prompt = "Password:"):
    if sys.platform.startswith('linux'):
        pswdfile = join(os.getenv("HOME") or '/home/Filin-GA',".pswd")
    if os.path.exists(pswdfile):
        return open(pswdfile,'r').read().strip()
    else:
        return getpass(prompt)

def nvl(value1, value2):
    return value1 if value1 else value2

def print_timestamp(stamp=''):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), stamp)

prts = print_timestamp

def cmove(src, dst):
    # if not 'WindowsError' in globals():
    #     WindowsError = IOError

    if os.path.exists(dst):
        os.remove(dst)
    try:
        shutil.move(src, dst)
    except (WindowsError, IOError) as e:
        print("cannot delete", e)

def cremove(filename):
    if os.path.exists(filename):
        try:
            os.remove(filename)
        except Exception:
            pass

def randstr(n):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(n))



def ds(val):
    if val.endswith('=\n'):
        try:
            return decodestring(val)
        except binascii.Error:
            return val
    else:
        return val

def time_to_secs(tme):
    tt = strptime(tme, '%H:%M:%S')
    return tt.tm_sec + tt.tm_min*60 + tt.tm_hour*3600

def makedirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def try_decode(val):
    def f(val):
        if isinstance(val, str):
            for code in ('UTF-8', 'WINDOWS-1251', 'cp866'):
                try:
                    return val.decode(code)
                except UnicodeDecodeError:
                    continue
        else:
            return val

    if isinstance(val, str):
        return f(val)
    if isinstance(val, list):
        return [f(v) for v in val]
    if isinstance(val, tuple):
        return tuple(f(v) for v in val)

    return f(val)

def utf(val):
    if isinstance(val, basestring):
        if isinstance(val, str):
            val = try_decode(val)
        return val.encode('UTF-8', 'ignore')
    else:
        return val

def win(val):
    if isinstance(val, basestring):
        if isinstance(val, str):
            val = try_decode(val)
        return val.encode('WINDOWS-1251', 'ignore')
    else:
        return val

def try_win(val):
    val = try_decode(val)
    if isinstance(val, unicode):
        return win(val)
    elif isinstance(val, (list, tuple)):
        return [win(v) for v in val]
    else:
        return val

def try_utf(val):
    val = try_decode(val)
    if isinstance(val, unicode):
        return utf(val)
    elif isinstance(val, (list, tuple)):
        return [utf(v) for v in val]
    else:
        return val

def printRec(fields, values):
    for k, v in zip(fields, values):
        print (k,'=',repr(v),try_decode(v), ',', type(v))

def normpath(filename): return filename.replace('/', '\\')
def unixpath(filename): return filename.replace('\\','/')

def is_rar(filename): return os.path.splitext(filename)[1].lower() == '.rar'
def is_xlsx(filename): return os.path.splitext(filename)[1].lower() == '.xlsx'
def is_xlsb(filename): return os.path.splitext(filename)[1].lower() == '.xlsb'
def is_csv(filename): return os.path.splitext(filename)[1].lower() == '.csv'
def is_xls(filename): return os.path.splitext(filename)[1].lower() == '.xls'
def is_zip(filename): return os.path.splitext(filename)[1].lower() == '.zip'
def is_pptx(filename): return os.path.splitext(filename)[1].lower() == '.pptx'

def rar_file(filename):
    rfilename = os.path.splitext(filename)[0] + '.rar'

    if os.path.exists(rfilename):
        os.remove(rfilename)

    call(["C:\\Program Files (x86)\\WinRAR\\Rar.exe", "a", "-ep", "-hpsber", rfilename, filename ])
    return rfilename

def unprotect_xlsx(filename, password):
    try:
        x = win32com.client.Dispatch('Excel.Application')
        x.DisplayAlerts = False
        x.Visible = False

        wb = x.Workbooks.Open(filename, False, True, None, ds(password))
        wb.Unprotect(ds(password))
        fn, ext = os.path.splitext(filename)

        newfn = normpath(join(dirname(filename), fn+"_unpass"+ext))
        # print "SaveAs: " + self.newfilename
        wb.SaveAs(newfn, None, '', '')
        wb.Close()
        x.Quit()
        x.Application.Quit()
        del x
        # os.remove(self.filename)
        # move(self.newfilename, self.filename)
    except pwt.com_error as e:
        x.Quit()
        del x
        print (e)


    return newfn if 'newfn' in locals() else None

def convert_xlsb_to_xlsx(filename):
    try:
        newfn = normpath(os.path.splitext(filename)[0] + '.xlsx')
        if os.path.exists(newfn):
            os.unlink(newfn)
        # x = win32com.client.Dispatch('Excel.Application')
        x = win32com.client.gencache.EnsureDispatch('Excel.Application')
        x.DisplayAlerts = False
        x.Visible = False

        wb = x.Workbooks.Open(filename, False, True, None)

        wb.SaveAs(newfn, FileFormat=win32com.client.constants.xlWorkbookDefault)
        wb.Close()
        x.Quit()
        x.Application.Quit()
        del x
        return newfn

    except pwt.com_error as e:
        x.Quit()
        del x
        print(e)
    except:
        if 'x' in locals():
            x.Quit()
            del x


def asktoresume(msg):
    res = ""
    while res not in ('Y','y','N', 'n'):
        res = raw_input(msg)
    return res in ['Y','y']

def compress_file(filename, zfilename):
    z = zipfile.ZipFile(zfilename, 'w', zipfile.ZIP_DEFLATED)
    z.write(filename, arcname=basename(filename), compress_type=zipfile.ZIP_DEFLATED)
    z.close()

    return zfilename


def upload(filename):
    if sys.platform.startswith('linux'):
        import smbclient
        sc = smbclient.SambaClient(server='Bronze1.ca.sbrf.ru', share='IK', username='Filin1-GA', password=GetPass(),
                                   domain='ALPHA',config_file='/home/Filin-GA/.smb/smb.conf')
        dest = "/IK0123/OUT/%s" % basename(filename)
        sc.upload(filename, dest)
    else:
        raise Exception('this function can only be called in linux platform')

class Authorization(object):
    SERVICE_NAME = 'iskra'
#    __slots__ = ['user', 'domain', 'mailbox', 'password', 'server', 'kr']
    def __init__(self, user, domain, mailbox, server):
        self.kr = keyring.get_keyring()
        self.user = user
        self.domain = domain
        self.mailbox = mailbox
#        self.password = self.get_password()
        self.server = server

    def get_password(self):
        p = self.kr.get_password(self.SERVICE_NAME, self.user)
        if not p:
            raise Exception("No password!")
            # p = getpass('Enter password for user "%s"' % self.user)
            # self.kr.set_password(self.SERVICE_NAME, self.user, p)
        return p

    @property
    def username(self):
        return '%s\\%s' % (self.domain, self.user)

    @property
    def password(self):
        return self.get_password()

ServAuth = Authorization(user='',
                         domain='',
                         mailbox='',
                         server="")

DKKAuth  = Authorization(user='i',
                         domain='',
                         mailbox='',
                         server="Outlook.ca.sbrf.ru")

ISKRACVMAuth = Authorization(user='',
                         domain='',
                         mailbox='',
                         server='Outlook.ca.sbrf.ru')


#import logging
#logging.getLogger("exchange").setLevel(logging.DEBUG)

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


class MailReceiver(object):
    savepath = join(workdir, "New")
    subject = None
    filter_date = date.today().isoformat()
    sleeptime = 30

    def __init__(self, password=None, savepath=None, auth_class=ISKRACVMAuth):

        self.pas = password or self.AuthClass.password
        self.savepath = savepath or join(workdir, "New")
        self.AuthClass = auth_class
        # self.creds = Credentials(username="", password=base64.decodestring(self.pas))
        self.creds = Credentials(username=self.AuthClass.username, password=ds(self.pas))
        #"cab-vop-mbx1035.omega.sbrf.ru"
        #"9016-mailcas.sbcorp.ub.sbrf.ru"
        #"cab-vop-mbx2028.omega.sbrf.ru"
        #"iskra_cvm@mail.ca.sbrf.ru"

        self.config = Configuration(server=self.AuthClass.server,
                                    credentials = self.creds,
                                    auth_type=NTLM,
                                    verify_ssl=False)

        self.account = Account( primary_smtp_address=self.AuthClass.mailbox,
                                config = self.config,
                                credentials = self.creds,
                                autodiscover = False,
                                access_type = DELEGATE)

        makedirs(self.savepath)

        self.__files = []
        self.__msgs = []

    def __saveToFile(self, filename, content):
        #fn = filename.encode("WINDOWS-1251") if isinstance(filename, unicode) else filename
        if is_rar(filename) or is_xls(filename) or is_xlsx(filename) or is_zip(filename) or is_xlsb(filename) or is_csv(filename):
            fn = filename
            print(fn)
            f = open(fn, 'wb')
            f.write(content)
            f.close()
            return True
        else:
            return False

    def isEmailValuable(self, email_addr):
        if self.author is None:
            return True
        for a in self.author:
            if email_addr.lower().find(try_win(a).lower()) > -1:
                return True
        return False

    def isMsgHasValuableEmail(self, msg):
        if not hasattr(msg, 'author'):
            return False
        else:
            return self.isEmailValuable(msg.author.email_address)

    def extractFilesFromMsg(self, msg, savepath):
        lst = []
        if msg.attachments:
            for a in msg.attachments:
                if isinstance(a, FileAttachment):
                    filename = normpath(join(savepath, basename(a.name)))

                    if self.__saveToFile(filename, a.content):
                        lst.append( filename )
        return lst

    def __receiveFile(self):
        res = []

        params = {}
        if self.filter_date:
            params['datetime_received__gt'] = self.filter_date
        if self.subject:
            params['subject__contains'] = try_win(self.subject)

        msgs = []

        print(params)
        print(self.author)

        if True:
            res = []
            for msg in self.account.inbox.filter( **params ).only('author','attachments').iterator():
                if not self.isMsgHasValuableEmail(msg):
                    continue
                msgs.append(msg)
                res += self.extractFilesFromMsg(msg, self.savepath)

        self.__msgs = list(set(msgs))
        return list(set(res))

    def showFriends(self, filter_date=None, subject=None, author=None):

        params = {}
        if filter_date:
            params['datetime_received__gt'] = filter_date.isoformat()
        if subject:
            params['subject__contains'] = subject
        if author:
            params['author__contains'] = author


        msgs = []
        if True:
            print("privedidruga:")
            for msg in self.account.root.get_folder_by_name('Приведи Друга').filter(**params).only('subject', 'body','author','datetime_sent', 'is_read').iterator():
                msgs.append(msg)

        return msgs

    def showMOMENTUM(self, filter_date=None, subject=None, author=None):

        params = {}
        if filter_date:
            params['datetime_received__gt'] = filter_date.isoformat()
        if subject:
            params['subject__contains'] = subject
        if author:
            params['author__contains'] = author


        msgs = []
        if True:
            print("momentum:")
            for msg in self.account.root.get_folder_by_name('MOMENTUM').filter(**params).only('subject', 'body','author','datetime_sent', 'datetime_received', 'is_read','attachments').iterator():
                msgs.append(msg)

        return msgs

    def showSMSANNA(self, filter_date=None, subject=None, author=None):

        params = {}
        if filter_date:
            params['datetime_received__gt'] = filter_date.isoformat()
        if subject:
            params['subject__contains'] = subject
        if author:
            params['author__contains'] = author


        msgs = []
        if True:
            print("smsanna:")
            for msg in self.account.root.get_folder_by_name('СМСАННА').filter(**params).only('subject', 'body','author','datetime_sent', 'datetime_received', 'is_read').iterator():
                msgs.append(msg)

        return msgs

    def showASISTOBE(self, filter_date=None, subject=None, author=None):

        params = {}
        if filter_date:
            params['datetime_received__gt'] = filter_date.isoformat()
        if subject:
            params['subject__contains'] = subject
        if author:
            params['author__contains'] = author


        msgs = []
        if True:
            print("asistobe:")
            for msg in self.account.root.get_folder_by_name('ASISTOBE').filter(**params).only('subject', 'body','author','datetime_sent', 'datetime_received','attachments','is_read').iterator():
                msgs.append(msg)

        return msgs

    def showDZO(self, filter_date=None, subject=None, author=None):

        params = {}
        if filter_date:
            params['datetime_received__gt'] = filter_date.isoformat()
        if subject:
            params['subject__contains'] = subject
        if author:
            params['author__contains'] = author


        msgs = []
        if True:
            print("dzo:")
            for msg in self.account.root.get_folder_by_name('ДЗО').filter(**params).only('subject', 'body','author','datetime_sent', 'datetime_received', 'attachments', 'is_read').iterator():
                msgs.append(msg)

        return msgs

    def showEFFKI(self, filter_date=None, subject=None, author=None):

        params = {}
        if filter_date:
            params['datetime_received__gt'] = filter_date.isoformat()
        if subject:
            params['subject__contains'] = subject
        if author:
            params['author__contains'] = author


        msgs = []
        if True:
            print("effki:")
            for msg in self.account.root.get_folder_by_name('EFFKI').filter(**params).only('subject', 'body','author','datetime_sent', 'datetime_received', 'attachments', 'is_read').iterator():
                msgs.append(msg)

        return msgs

    def showWIFI(self, filter_date=None, subject=None, author=None):

        params = {}
        if filter_date:
            params['datetime_received__gt'] = filter_date.isoformat()
        if subject:
            params['subject__contains'] = subject
        if author:
            params['author__contains'] = author


        msgs = []
        if True:
            print("wifi:")
            for msg in self.account.root.get_folder_by_name('CLIENTS').filter(**params).only('subject', 'body','author','datetime_sent', 'datetime_received', 'attachments', 'is_read').iterator():
                msgs.append(msg)

        return msgs


    def showPRPU(self, filter_date=None, subject=None, author=None):

        params = {}
        if filter_date:
            params['datetime_received__gt'] = filter_date.isoformat()
        if subject:
            params['subject__contains'] = subject
        if author:
            params['author__contains'] = author


        msgs = []
        if True:
            print("prpu05:")
            for msg in self.account.root.get_folder_by_name('ПУ ДЗО').filter(**params).only('subject', 'body','author','datetime_sent', 'datetime_received', 'attachments', 'is_read').iterator():
                msgs.append(msg)

        return msgs

    def showSMP(self, filter_date=None, subject=None, author=None):

        params = {}
        if filter_date:
            params['datetime_received__gt'] = filter_date.isoformat()
        if subject:
            params['subject__contains'] = subject
        if author:
            params['author__contains'] = author


        msgs = []
        if True:
            print("sm_podruga:")
            for msg in self.account.root.get_folder_by_name('SM_PODRUGA').filter(**params).only('subject', 'body','author','datetime_sent', 'datetime_received', 'is_read', 'attachments').iterator():
                msgs.append(msg)

        return msgs

    def showBSPS(self, filter_date=None, subject=None, author=None):

        params = {}
        if filter_date:
            params['datetime_received__gt'] = filter_date.isoformat()
        if subject:
            params['subject__contains'] = subject
        if author:
            params['author__contains'] = author


        msgs = []
        if True:
            print("bis_spasibo:")
            for msg in self.account.root.get_folder_by_name('Бизнес СПАСИБО').filter(**params).only('subject', 'body','author','datetime_sent', 'datetime_received', 'is_read').iterator():
                msgs.append(msg)

        return msgs

    def showCRMMONITOR(self, filter_date=None, subject=None, author=None):

        params = {}
        if filter_date:
            params['datetime_received__gt'] = filter_date.isoformat()
        if subject:
            params['subject__contains'] = subject
        if author:
            params['author__contains'] = author


        msgs = []
        if True:
            print("crm_monitor:")
            for msg in self.account.root.get_folder_by_name('CRM Monitor').filter(**params).only('subject', 'body','author','datetime_sent','attachments', 'datetime_received', 'is_read').iterator():
                msgs.append(msg)

        return msgs
    
    def showTMS(self, filter_date=None, subject=None, author=None):

        params = {}
        if filter_date:
            params['datetime_received__gt'] = filter_date.isoformat()
        if subject:
            params['subject__contains'] = subject
        if author:
            params['author__contains'] = author


        msgs = []
        if True:
            print("teradata_monitor_serv:")
            for msg in self.account.root.get_folder_by_name('TeradataMonitorServ').filter(**params).only('subject', 'body','author','datetime_sent','attachments', 'datetime_received', 'is_read').iterator():
                msgs.append(msg)

        return msgs

    def showScripts(self, filter_date=None, subject=None, author=None):

        params = {}
        if filter_date:
            params['datetime_received__gt'] = filter_date.isoformat()
        if subject:
            params['subject__contains'] = subject
        if author:
            params['author__contains'] = author


        msgs = []
        if True:
            print("scripts:")
            for msg in self.account.root.get_folder_by_name('Скрипты').filter(**params).only('subject', 'body','author','datetime_sent','attachments', 'datetime_received', 'is_read').iterator():
                msgs.append(msg)

        return msgs

    def showMessages(self, filter_date=None, subject=None, author=None):

        params = {}
        if filter_date:
            params['datetime_received__gt'] = filter_date.isoformat()
        if subject:
            params['subject__contains'] = subject
        if author:
            params['author__contains'] = author


        msgs = []
        if False:
            print("privedidruga:")
            for msg in self.account.root.get_folder_by_name('Приведи Друга').filter(**params).only('subject', 'body','author','datetime_sent', 'is_read').iterator():
                msgs.append(msg)
        
        if True:
            print("inbox:")
            for msg in self.account.inbox.filter(**params).only('subject', 'body', 'is_read').iterator():
                msgs.append(msg)
                # print '', msg.author.email_address, ',', msg.subject, ',', msg.datetime_received, ',', len(msg.attachments)

        if False:
            print('sent:')
            for msg in self.account.sent.filter().only('subject', 'body', 'is_read').iterator():
                msgs.append(msg)
                # print '', msg.author.email_address if hasattr(msg,'author') else None, ',', msg.subject, ',', msg.datetime_received, ',', len(msg.attachments)

        if False:
            print('junk:')
            for msg in self.account.junk.filter().only('subject', 'body', 'is_read').iterator():
                msgs.append(msg)
                # print '', msg.author.email_address, ',', msg.subject, ',', msg.datetime_received, ',', len(msg.attachments)

        if True:
            print('trash:')
            for msg in self.account.trash.filter().only('subject', 'body', 'is_read').iterator():
                msgs.append(msg)
                # print '', msg.subject, ',', msg.datetime_received, ',', len(msg.attachments)

        return msgs
    
    def checkMessages(self):
        new_messages=pd.DataFrame()
        for i,f in enumerate(['Входящие','Екатерина','Иван','Марсель','Алексей','Кирилл']):
            msgs = []
            for msg in self.account.root.get_folder_by_name(f).filter().only('is_read').iterator():
                try:
                    if not msg.is_read:
                        msgs.append(msg)
                except:
                    pass
            new_messages=new_messages.set_value(i,'Folder',f)
            new_messages=new_messages.set_value(i,'New Messages',len(msgs))
        new_messages=new_messages.set_index('Folder')
        return new_messages
    
    def getAccepts(self, filter_date=None, subject=None, author=None):
        params = {}
        if filter_date:
            params['datetime_received__gt'] = filter_date.isoformat()
        if subject:
            params['subject__contains'] = subject
        if author:
            params['author__contains'] = author

        params['is_read'] = False

        msgs = []

        for msg in self.account.inbox.filter(**params).only('is_read', 'author', 'subject', 'datetime_received', 'body').iterator():
            if not msg.is_read:
                msgs.append(msg)

        return msgs


#    def clearAll(self):
#        for msg in self.showMessages(filter_date=date(2017,1,1)):
#            msg.delete()
#
#        print("All Messages deleted")

    def startReceive(self, tries=5, filter_date=None, subject=None, author=None):

        self.filter_date = filter_date or date.today()
        if subject: self.subject = subject
        if author:
            self.author = author if isinstance(author, (tuple,list)) else (author,)
        else:
            self.author = (None,)

        if isinstance(self.filter_date, date):
            self.filter_date = self.filter_date.isoformat()

        while tries > 0:
            print("try to search msg", datetime.now().isoformat(), filter_date)
            tries -= 1
            files = self.__receiveFile()
            if (files):
                print("OK")
                return files
            print("failed")
            if tries > 0: sleep(self.sleeptime)
        return []

#    def deleteMessages(self):
#        for m in self.__msgs:
#            m.delete()

    # def sendMessage(self, recipients, theme, files):
    #
    #     m = Message(
    #         account = self.account,
    #         subject = theme,
    #         body = u"Ôàéë ñ îøèáêàìè",
    #         to_recipients = [Mailbox(email_address=r) for r in recipients],
    #     )
    #
    #     for f in files:
    #         fa = FileAttachment(name=basename(f).decode("WINDOWS-1251"), content=open(f,'rb').read())
    #         m.attach(fa)
    #
    #     m.send_and_save()

    def sendMessageCopy(self, recipients, cc_recs, theme, body, files=[]):

        m = Message(
            account = self.account,
            folder = self.account.sent,
            subject = theme,
            body = body,
            to_recipients = [Mailbox(email_address=r) for r in recipients],
            cc_recipients = [Mailbox(email_address=r) for r in cc_recs]
        )

        for f in files:
            fa = FileAttachment(name=basename(f), content=open(f,'rb').read())
            m.attach(fa)

        m.send_and_save()

    def sendMessage(self, recipients, theme, body, files=[]):

        m = Message(
            account = self.account,
            folder = self.account.sent,
            subject = theme,
            body = body,
            to_recipients = [Mailbox(email_address=r) for r in recipients],
        )

        for f in files:
            fa = FileAttachment(name=basename(f), content=open(f,'rb').read())
            m.attach(fa)

        m.send_and_save()
    
    def sendHtmlMessage(self, recipients, theme, body, files=[], cc_recs=[]):

        m = Message(
            account = self.account,
            folder = self.account.sent,
            subject = theme,
            body = HTMLBody(body),
            to_recipients = [Mailbox(email_address=r) for r in recipients],
            cc_recipients = [Mailbox(email_address=r) for r in cc_recs]
        )

        for f in files:
            fa = FileAttachment(name=basename(f), content=open(f,'rb').read())
            m.attach(fa)

        m.send_and_save()
    
    def getTenders(self,destination):
        for msg in self.account.root.get_folder_by_name('Банковские гарантии').filter().only('attachments').iterator():
            self.extractFilesFromMsg(msg,destination)
            msg.delete()

    def getPilots(self, filter_date, author=None):
        # params = {}
        msgs = []
        params = {'datetime_received__gt': filter_date.isoformat()}

        if author is None:
            self.author = author
        elif not isinstance(author, (tuple,list)):
            self.author = [author]
        else:
            self.author = author

        for msg in self.account.inbox.filter( **params ).only('author','attachments','subject', 'body','is_read').iterator():
            if isinstance(msg, Message):
                if not self.isEmailValuable(msg.author.email_address):
                    continue
                if msg.attachments:
                    msgs.append(msg)

        return msgs

                # for a in msg.attachments:
                #     if isinstance(a, FileAttachment):
                #         filename = normpath(join(self.savepath, a.name))
                #
                #         if self.__saveToFile(filename, a.content):
                #             res.append(filename)
