"""
# module helps to read information from google tables
# @author: vndanilchenko@gmail.com
"""

import httplib2
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import json
import os
import googleapiclient.discovery

# подготовим конфиг для доступа к гугл таблицам
GOOGLE_SHEET_PK_ID = os.environ.get('GOOGLE_SHEET_PK_ID')
GOOGLE_SHEET_PK = os.environ.get('GOOGLE_SHEET_PK')
# локально возвращает путь к файлу к ключем
if 'PRIVATE' not in GOOGLE_SHEET_PK:
    with open(os.environ.get('GOOGLE_SHEET_PK')) as f:
        GOOGLE_SHEET_PK = f.read()
with open(r'./cfg/read_schedule_config.json') as f:
    CREDENTIALS_FILE = f.read()
CREDENTIALS_FILE = json.loads(CREDENTIALS_FILE)
CREDENTIALS_FILE.update({'private_key_id': GOOGLE_SHEET_PK_ID, 'private_key': GOOGLE_SHEET_PK})


class Read_google_sheet_schedule:

    def __init__(self):
        # Файл, полученный в Google Developer Console
        self.CREDENTIALS_FILE = CREDENTIALS_FILE

        # Авторизуемся и получаем service — экземпляр доступа к API
        self.credentials = ServiceAccountCredentials.from_json_keyfile_dict(self.CREDENTIALS_FILE,
            ['https://www.googleapis.com/auth/spreadsheets',
                 'https://www.googleapis.com/auth/drive']
            )
        print('инициализация Read_google_sheet_schedule')

    def read_data(self, data_range, sheet_id):
        """
        :param data_range:
        :param sheet_id:
        :return:
        """
        # чтение из файла
        values = self.service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range=data_range,
            majorDimension='ROWS'
        ).execute()
        # pprint(values)
        df = pd.DataFrame(values['values'])
        df.columns = df[df.index == 0].values.tolist()[0]
        df.drop([0], axis=0, inplace=True)
        return df

    # функция делает чтение из разных листов/диапазонов в зависимости от команды
    def run(self, command):
        """
        :param command:
        :return:
        """
        self.httpAuth = self.credentials.authorize(httplib2.Http())
        self.service = googleapiclient.discovery.build('sheets', 'v4', http = self.httpAuth)
        if command=='get_schedule':
            return self.read_data(data_range='A1:J1000', sheet_id='1mTxBPHze-kuSmMl6fTb_EGO3LI8Q0rZEiY-VUBccGIA')
        elif command=='who_is':
            return self.read_data(data_range='A1:D1000', sheet_id='1mTxBPHze-kuSmMl6fTb_EGO3LI8Q0rZEiY-VUBccGIA')
        elif command=='get_responses':
            return self.read_data(data_range='A1:F1000', sheet_id='1XIR4CLXj_vRgv04_MF18JBMG4RITTP28RQ0ygNUNtGk')
        else:
            return pd.DataFrame()

# if __name__ == '__main__':
#     obj = Read_google_sheet_schedule()
#     df = obj.run(command='get_responses')
#     df.columns
#     df[df['class']=='0']