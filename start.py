"""
# slack-bot for Netology
# @author: vndanilchenko@gmail.com

# usage example:
    import os
    import slack

    rtmclient = slack.RTMClient(token=slack_token)

    def say_hello(**kwargs):
        data = kwargs['data']
        print(data)
        text = data.get('text', [])

        if 'Hello' in text:
            channel_id = data['channel']
            thread_ts = data['ts']
            user = data['user']

            webclient = slack.WebClient(slack_token)
            webclient.chat_postMessage(
                channel=channel_id,
                text="Hi <@{}>!".format(user),
                thread_ts=thread_ts
            )

    rtmclient.on(event='message', callback=say_hello)
    rtmclient.start()
"""
import json
import os
import requests
import slack
import models as models
import read_google_sheets as schedule
import send_email as email
import pandas as pd
import re
from datetime import datetime, date
import time

# импортируем токены
try:
    token = os.environ.get('SlACK_TOKEN')
except:
    from private.token import token
try:
    weather_key = os.environ.get('WEATHER_TOKEN')
except:
    from private.token import weather_key


THRESHOLD_KB = 0.6           # считаем, что знаем вопрос
THRESHOLD_EXPLICIT = 0.8     # граница уверенного ответа
THRESHOLD_APPROVAL = 0.6     # граница вопроса с уточнением
THRESHOLD_YES_NO = 0.8       # порог отсечки при положительном ответе на уточнение
THRESHOLD_USER_QUIT = 0.8    # порог отсечки при проверке на отмену ввода параметров при режиме query


# скомпилим регулярочки
# <mailto:vndanilchenko@gmail.com|vndanilchenko@gmail.com>
pattern_mailto = re.compile(r'\<{1}[a-zA-Z0-9\:\@\.]{1,}\|{1}')
pattern_id = re.compile(r'^(\<{1}\@{1}[a-zA-Z0-9]{1,}\>{1})')
# <#CMX9X7F37|chat-bot-in-education>
pattern_channel1 = re.compile(r'\<{1}\#{1}[A-Z0-9]{1,}\|{1}')
pattern_rbracket = re.compile(r'\>')
pattern_email = re.compile(r'[0-9a-zа-я_]{1,}\@{1}[a-z]{1,}\.{1}[a-z]{1,}')


class Class_none(Exception):
    pass

class Slackbot:

    def __init__(self):
        self.slack_token = token
        self.rtmclient = slack.RTMClient(token=self.slack_token)
        self.webclient = slack.WebClient(token=self.slack_token)
        self.channels = None
        self.thread_ts = None
        self.user = None
        self.text_in = None
        self.text_out = None
        self.schedule = schedule.Read_google_sheet_schedule()
        self.responses_df = self.schedule.run(command='get_responses')
        self.commands = ['send_info', 'show_weather', 'get_schedule', 'tell_joke', 'send_email', 'who_is']
        self.classifier = models.Classifier()
        self.state_in = {'intent': None, 'command': None, 'state': 'normal', 'reply': None, 'query_params': {}}
        self.state_out = {'intent': None, 'command': None, 'state': 'normal', 'reply': None, 'query_params': {}}
        self.send_email = email.Send_email()
        self.weather_descr = pd.read_csv('./data/Multilingual_Weather_Conditions.csv', sep=',', encoding='utf-8')
        self.sleep_cnt = 0
        self.time_state = 'normal'

    def set_default_state(self):
        """
        октат состояний
        :return:
        """
        self.state_out = {'intent': self.state_in['intent'], 'command': self.state_in['command'], 'state': 'normal', 'reply': self.state_in['reply'], 'query_params': {}}

    def get_channel_id(self, channel_name):
        """
        функция ищет среди каналов заданное название и возвращает id, либо None
        :param channel_name:
        :return:
        """
        req = self.webclient.conversations_list(types="public_channel, private_channel")
        for chanel in req.data['channels']:
            # print(chanel)
            if chanel['name'] == channel_name:
                return chanel['id']
        else:
            return None

    def detect_message(self, **kwargs):
        """
        основная функция - если получает сообщения из нужного канала и эти сообщения требуют ответа,
        то вызывает по очереди все нужные дл получения ответа функции
        :param kwargs:
        :return:
        """
        try:
            data = kwargs['data']
            self.text_in = data.get('text', [])
            if self.text_in and data.get('subtype', []) not in ['bot_message']:
                self.web_client = kwargs.get('web_client', [])
                self.rtm_client = kwargs.get('rtm_client', [])
                self.channels = [data['channel']]
                if 'blocks' in data:
                    self.direct_link = True if data['blocks'][0]['elements'][0]['elements'][0]['type']=='user' else False
                else:
                    self.direct_link = False
                self.thread_ts = data['ts']
                self.user = data['user']
                if ('DMQS540H1' in self.channels or self.direct_link) and 'client_msg_id' in data:
                    print('text_in: ', self.text_in)
                    self.state_in = self.state_out
                    self.text_out = self.make_response()
                    self.reply()
                    print('state_in: ', self.state_in)
                    print('state_out: ', self.state_out)
                    print('{:*^170}'.format(''))
            # если вдруг начнет крэшиться отловить некоторые ошибки и пропустить
        except Class_none as e:
            print('{: ^50}'.format('+ + +'))
            print('надо добавить интент в список реакций на google docs! \nerror: ', e.args)
            print('{: ^50}'.format('+ + +'))
        except TimeoutError as e:
            print('{: ^50}'.format('+ + +'))
            print(self.catch_problems(), ' \nerror: ', e.args)
            if self.sleep_cnt < 3:
                print('попробуем повторно подключиться через 30 секунд')
                time.sleep(30)
            elif self.sleep_cnt < 10:
                print('попробуем повторно подключиться через 60 секунд')
                time.sleep(60)
            elif self.sleep_cnt < 20:
                print('попробуем повторно подключиться через 5 минут')
                time.sleep(300)
            elif self.sleep_cnt < 30:
                print('попробуем повторно подключиться через 10 минут')
                time.sleep(600)
            elif self.sleep_cnt < 40:
                print('попробуем повторно подключиться через 30 минут')
                time.sleep(1800)
            else:
                print('попробуем повторно подключиться через 60 минут')
                time.sleep(3600)
            self.run()
            print('{: ^50}'.format('+ + +'))
        except Exception as e:
            print('{: ^50}'.format('+ + +'))
            print(self.catch_problems(), '\nerror: ', e.args)
            print('{: ^50}'.format('+ + +'))

    def get_query_params(self):
        """
        функция проверяет заполненность параметров
        :return:
        """
        print('зашли в query')
        # выбирам нужный интент
        if self.state_in['intent'] == "разослать информацию студентам на email":
            params = {'addr_to': 'для отправки введите имэйл',
                      'theme': 'введите тему сообщения',
                      'body': 'введите сообщение',
                      'ok': False}
        elif self.state_in['intent'] == "сделать уведомление студентам канала":
            params = {'channel_name': 'для отправки введите название канала',
                      'channel_id': 'это должно заполнитсья само после ввода названия канала',
                        'body': 'введите сообщение',
                        'ok': False}
        elif self.state_in['state']=='query':
            params = {'column_id': 'выберите цифрой какую информацию вы хотите посмотреть по преподавателю:\n ' \
                                    '1. образование\n ' \
                                    '2. должность\n ' \
                                    '3. дополнительно',
                        'ok': False}

        else:
            print(f"НЕ НАШЛИ ИНТЕНТ: {self.state_in['intent']}")
            return "к сожалению не смог обработать этот запрос"

        # проверка на ожидание ответа по параметру
        for param, val in self.state_in['query_params'].items():
            if val == 'waiting':
                self.text_in = re.sub(pattern_id, '', self.text_in.strip())
                print('в ожидании заполнения команды: {}'.format(param))
                # проверка на формат почты
                if param == 'addr_to':
                    prep_email = re.sub(pattern_mailto, '', self.text_in.strip())
                    prep_email = re.sub(pattern_rbracket, '', prep_email)
                    test_email = re.sub(pattern_email, '', prep_email)
                    print('email: ', prep_email)
                    if test_email:
                        self.state_out = self.state_in
                        return f'параметр не распознан:\n {params[param]}'
                    else:
                        self.text_in = prep_email
                        self.state_in['query_params']['addr_to'] = prep_email
                # проверка названия канала без хэштега и наличия канала вообще
                elif param == 'channel_name':
                    test_channel = re.sub(pattern_channel1, '', self.text_in)
                    test_channel = re.sub(pattern_rbracket, '', test_channel)
                    self.text_in = test_channel.replace('#', '').strip()
                    # проверка наличия канала
                    channel = self.get_channel_id(channel_name=self.text_in)
                    if not channel:
                        self.repeat_query(params[param])
                    else:
                        self.state_in['query_params']['channel_id'] = channel
                        self.state_in['query_params'][param] = self.text_in
                # проверка наличия записи в зарпосе про преподавателя
                elif param == 'column_id':
                    # проверим не ввел ли пользователь несколько значений
                    variants = {'education': ['1', 'образование', 'один', 'единица', 'первое'],
                              'position': ['2', 'должность', 'два', 'двойка', 'второе'],
                              'extra': ['3', 'дополнительно', 'доп', 'три', 'тройка', 'третье']}
                    user_choice = set()
                    for elem in str(self.text_in).split(' '):
                        for j in range(len(variants)):
                            if elem in list(variants.values())[j]:
                                user_choice.add(list(variants.keys())[j])
                    if not user_choice:
                        # проверим на фразу "все"
                        if any(i in ['все', 'весь', 'всю', 'список'] for i in str(self.text_in).split(' ')):
                            [user_choice.add(i) for i in list(variants.keys())]
                        else:
                            self.repeat_query(params[param])
                    else:
                        self.state_in['query_params'][param] = list(user_choice)
                elif param == 'ok':
                    max_score, max_class = self.classifier.predict_proba(self.text_in, model_flag=1)
                    print('ПРОВЕРКА ДА/НЕТ: max_score: ', max_score, 'max_class', max_class)
                    if max_class == 1 and max_score > THRESHOLD_YES_NO:
                        self.state_in['query_params']['ok'] = True
                        print('прошли проверку на наличие подтвреждения на отправку - положительно')
                    elif max_class == 0 and max_score > THRESHOLD_YES_NO:
                        # выходим из режима при уверенном отказе
                        print('прошли проверку на наличие подтвреждения на отправку - негативно')
                        return self.realize_user_quit()
                else:
                    self.state_in['query_params'][param] = self.text_in
                # обрабатываем по одному параметру за раз
                break


        # оставим различия
        [params.pop(k) for k in self.state_in['query_params']]
        print('сверили параметры по интенту, осталось заполнить: {}'.format(len(params)))

        # не осталось параметров для заполнения
        if not params:
            print('прошли отсечку по трэшолду согласия на отправку')
            # channel = self.get_channel_id(channel_name='chat-bot-in-education')
            if self.state_in['intent'] == "разослать информацию студентам на email":
                response = self.send_email.run(addr_to=self.state_in['query_params']['addr_to'],
                                               theme=self.state_in['query_params']['theme'],
                                               body=self.state_in['query_params']['body'])
            elif self.state_in['intent'] == "сделать уведомление студентам канала":
                members = self.web_client.conversations_members(channel=self.state_in['query_params']['channel_id'])
                self.channels = members.data['members']
                print('персональное сообщение всем в канале "{}" отправлено'.format(self.state_in['query_params']['channel_name']))
                response = self.state_in['query_params']['body']
            elif self.state_in['intent'] == 'узнать о преподавателе':
                # проверим не ввел ли пользователь несколько значений
                print('пользователь сделал выбор: ', self.state_in['query_params']['column_id'])
                print('команда: ', self.state_in['command'])
                temp_df = self.schedule.run(command=self.state_in['command'])
                temp_df = temp_df[self.state_in['query_params']['column_id'] + ['teacher']]
                print(temp_df.columns)
                response = 'информация по преподавателям: '
                # temp_df = pd.DataFrame(columns=['test', 'test2'], index = [0,1], data=[[1,2], [3,4]])
                for i in temp_df.iterrows():
                    for j in range(len(i)):
                        response += '\n{}'.format(i[j])
                print(response)
                # response = temp_df[self.state_in['query_params']['column_id']+['teacher']]
                print('отправил данные: ')
            else:
                print('ПОПЫТКА ОТПРАВИТЬ БЕЗ ИНТЕНТА В СПИСКЕ: <<{}>>'.format(self.state_in['intent']))
                response = 'unknown command'
            self.set_default_state()
            print('отправили')

        # заполнение параметров
        else:
            param = list(params.keys())[0]
            if param == 'ok':
                print('последний параметр для заполнения')
                self.state_out = self.state_in
                print(f"уточняем намерение {self.state_in['intent']}")
                response = f"вы уверены, что хотите {self.state_in['intent']} с параметрами:\n"
                for p in self.state_in['query_params']:
                    response += "{:+^20}\n".format(' '+ p +' ')
                    response += f"<<{self.state_in['query_params'][p]}>>\n"
                response += "? (да/нет)"
            else:
                print(f'заполняем параметр {param}')
                self.state_out = self.state_in
                response = params[param]
            self.state_out['query_params'][param] = 'waiting'
        return response

    def return_intent(self, max_class):
        """
        принимает метку класса и возвращает название интента
        :param max_class:
        :return:
        """
        if str(max_class) not in self.responses_df['class'].tolist():
            raise Class_none('полученного класса нет в документе гугл')
        intent_line = self.responses_df[self.responses_df['class']==str(max_class)]
        intent = intent_line.intent.values[0]
        command = intent_line.command.values[0]
        print('команда in: ', command)
        command = None if command not in self.commands else command
        print('команда out: ', command)
        # проверим наличие особого времени
        print('text_in: ', self.text_in)
        self.time_state = self.classifier.check_time(self.text_in)
        print('получили состояние времени: ', self.time_state)
        if self.time_state == 'future':
            reply = intent_line.reply_future.values[0]
        elif self.time_state == 'past':
            reply = intent_line.reply_past.values[0]
        else:
            reply = intent_line.reply_now.values[0]
        print('прошел intent, определил время: ', self.time_state)
        return intent, command, reply

    def convert_datetime(self, temp_date, temp_time):
        """
        преобразует строковый формат даты и времени из разных полей в одну переменную datetime
        :param temp_date:
        :param temp_time:
        :return:
        """
        # return (datetime.strptime(str(temp_date), '%Y-%m-%d %H:%M:%S') + (datetime.combine(date.min, temp_time) - datetime.min))
        return (datetime.strptime(str(temp_date), '%m/%d/%Y') + (datetime.combine(date.min, datetime.strptime(str(temp_time), '%H:%M').time()) - datetime.min))

    def do_command(self, command, reply):
        """
        выполняет определенную команду, если такая присутствует в интенте
        :param command:
        :param reply:
        :return:
        """
        response = reply
        # проверка на наличие команды
        if command == 'get_schedule':
            print('зашел в команду: ', command)
            temp_df = self.schedule.run(command=command)
            print('зашел в интент')
            temp_df['datetime'] = temp_df.apply(lambda row: self.convert_datetime(row.date, row.time), axis=1)
            # print(temp_df.head(1))
            temp_response = temp_df[temp_df['datetime'] > datetime.today().strftime('%m/%d/%Y %H:%M:%S')].sort_values(by='datetime')[['datetime', 'theme', 'subtheme', 'location', 'teachers', 'duration']].iloc[0,:].copy()
            response = f"следующее занятие будет: {temp_response[temp_response.index == 'datetime'].values[0]}, адрес: {temp_response[temp_response.index == 'location'].values[0]}, " \
                       f"блок: {temp_response[temp_response.index == 'theme'].values[0]}, тема занятия: {temp_response[temp_response.index == 'subtheme'].values[0]}, " \
                       f"преподаватели: {temp_response[temp_response.index == 'teachers'].values[0]}, продолжительность: {temp_response[temp_response.index == 'duration'].values[0]} {'час' if temp_response[temp_response.index == 'duration'].values[0] == 1 else 'часа'}"
            print(f'ответ по команде: {response}')
            # temp_response = None
            # TODO: сделать проверку на наличие урока в текущий момент (надо ввести в excel продолжительность занятия)
        elif command == 'tell_joke':
            print('зашел в команду: ', command)
            url = 'http://rzhunemogu.ru/RandJSON.aspx?CType=1'
            try:
                result = json.loads(requests.get(url).text, strict=False)
                response = result['content']
            except Exception as e:
                print(self.catch_problems(self), ' команда:', command, ' ошибка: ', e.args)
                # self.do_command(self, 'tell_joke', response)
            print(f'ответ по команде: {response}')
        elif command == 'show_weather':
            print('зашел в команду: ', command)
            url = 'http://api.weatherstack.com/current'
            params = {
                'access_key': weather_key,
                'query': 'Moscow'
                # ,'language' : 'Russian'
            }
            result = json.loads(requests.get(url, params).text, strict=False)
            print(result['current'])
            response = f"{result['current']['weather_icons']}, {result['location']['name']} : {self.weather_descr[(self.weather_descr.lang_name == 'Russian') & (self.weather_descr.overhead_code == result['current']['weather_code'])].trans_text_day.values[0]}" \
                            f"температура: {result['current']['temperature']}, скорость ветра: {result['current']['wind_speed']}, атмосферное давление: {result['current']['pressure']}"
            print(f'ответ по команде: {response}')
            # TODO: переименовать все атрибуты и отобрать только нужные (город, температура, ветер, давление, описание погоды), разобраться как показать иконку погоды
        elif command == 'send_email' or command == 'send_info' or command == 'who_is':
            print('зашел в команду: ', command)
            self.state_in = self.state_out
            self.state_in['state'] = 'query'
            response = self.get_query_params()
            print(f'ответ по команде: {response}')
        else:
            print('НЕОПИСАННАЯ КОМАНДА!')
            response = 'без команды зашел: {}'.format(self.commands)
        return response

    def repeat_query(self, param):
        """
        повтор запроса на получение информации, если пользователь не заполнил параметр
        :param param:
        :return:
        """
        self.state_out = self.state_in
        return f'параметр не распознан:\n {param}'

    def catch_problems(self):
        """
        если что-то пошло не так
        :return:
        """
        self.set_default_state()
        return 'что-то пошло не так..'

    def realize_user_quit(self):
        """
        если пользователь отказался подтверждать выполнение скрипта
        :return:
        """
        self.set_default_state()
        return 'хорошо, команда отменена.\nвсегда можете попробовать сначала ;)'

    def terminal_response(self):
        """
        возвращает стандартное сообщение, если скор не проходит трэшолд
        :return:
        """
        self.set_default_state()
        return f"<@{self.user}>, \nя не понял, попробуй перефразировать"

    def negative_response(self):
        """
        возвращает стандартное сообщение, если ответ клиента НЕТ
        :return:
        """
        self.set_default_state()
        return f"<@{self.user}>, \nвозможно я еще смогу тебе помочь, попробуй иначе поставить вопрос"

    def make_response(self):
        """
        функция проверяет запрос на прохождение отсечек и режим
        :return:
        """
        # model = check_kb.recognize_kb()
        # model.predict_proba('привет как жизнь чем занимаешься работаешь или учишься', True)
        if self.state_in['state'] == 'approval':
            print('APPROVAL')
            # # проверка классификаторов ДА/НЕТ и отсечку по трэшолду
            max_score, max_class = self.classifier.predict_proba(self.text_in, model_flag=1)
            print(f"THRESHOLD_APPROVAL пройден {max_score}" if max_score > THRESHOLD_APPROVAL else f"запрос не в базе знаний {max_score}")
            if max_class==1 and max_score > THRESHOLD_YES_NO:
                print('прошли треэшолд APPROVAL, класс: ', max_class)
                if self.state_in['command']:
                    print('есть ответ по команде: {}'.format(self.state_in['command']))
                    reply = self.do_command(self.state_in['command'], self.state_in['reply'])
                else:
                    reply = self.state_in['reply']
                self.state_out = {'intent': self.state_in['intent'], 'command': self.state_in['command'], 'state': 'normal', 'reply': reply, 'query_params': {}}
                # return reply
            else:
                self.set_default_state()
                reply = self.negative_response()
        elif self.state_in['state'] == 'query':
            # проверим не захотел ли пользователь сбросить режим
            max_score, max_class = self.classifier.predict_proba(self.text_in, model_flag=2)
            if max_class==1 and max_score > THRESHOLD_USER_QUIT:
                # выходим из режима
                reply = self.realize_user_quit()
            else:
                # функция обработает входной интент и проверит на заполнение всех атрибутов
                print('получили признак query, перенаправили сообщение в функцию')
                reply = self.get_query_params()

        else:
            # prediction_kb = self.recognize_kb.kb_predict_proba(self.text_in)
            max_score, max_class = self.classifier.predict_proba(self.text_in, model_flag=0)
            print(f"база распознана {max_score}" if max_score > THRESHOLD_KB else f"запрос не в базе знаний {max_score}")
            print('max_class: ', max_class)
            if max_score > THRESHOLD_KB: # Был prediction_kb
                # return f"Hi <@{self.user}>! \n I know what you mean.."
                # max_score, max_class = self.recognize_kb.intent_predict_proba(self.text_in)
                print('прошли треэшолд KB, класс: ', max_class)
                print('тип класса: ', type(max_class))
                intent, command, reply = self.return_intent(max_class)
                print('--time_state', self.time_state)
                self.state_out = {'intent': intent, 'command': command, 'state': 'normal', 'reply': reply, 'query_params': {}}
                print('нашел интент {}, команду {}, скор {}'.format(intent, command, max_score))
                if max_score > THRESHOLD_EXPLICIT:
                    print('EXPLICIT')
                    print(f'прошел отсечку по скору')
                    if command:
                        print('есть ответ по команде: {}'.format(command))
                        reply = self.do_command(command, reply)
                    # return reply
                elif max_score > THRESHOLD_APPROVAL:
                    self.state_out = {'intent': intent, 'command': command, 'state': 'approval', 'reply': reply, 'query_params': {}}
                    translation_states = {'normal': '', 'future': 'в будущем', 'past': 'в прошлом', 'error': ''}
                    reply = 'ты хочешь {}({})?'.format(intent, translation_states[self.time_state])

                else:
                    reply = self.terminal_response()
                    self.set_default_state()
            else:
                reply = self.terminal_response()
                self.set_default_state()
        return reply

    def reply(self):
        """
        возврат ответа
        :return:
        """
        self.time_state = 'normal'
        self.sleep_cnt = 0
        for channel in self.channels:
            self.web_client.chat_postMessage(
                channel=channel
                ,text=self.text_out # + " " + str(self.direct_link)# str(self.channels)
                # thread_ts=self.thread_ts
                ,as_user='UN1S55TNY'
            )

    def run(self):
        """
        слушает события
        :return:
        """
        print('{:+^50}'.format(' START '))
        self.rtmclient.on(event='message', callback=self.detect_message)
        self.rtmclient.start()


if __name__ == '__main__':
    bot = Slackbot()
    bot.run()
