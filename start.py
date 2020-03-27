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
import utils.models as models
import read_google_sheets as schedule
import send_email as email
import pandas as pd
import re
from datetime import datetime, date

# импортируем токены
try:
    token = os.environ.get('SlACK_TOKEN')
except:
    from private.token import token
try:
    weather_key = os.environ.get('WEATHER_TOKEN')
except:
    from private.token import weather_key



TRESHOLD_KB = 0.6           # считаем, что знаем вопрос
TRESHOLD_EXPLICIT = 0.8     # граница уверенного ответа
TRESHOLD_APPROVAL = 0.6     # граница вопроса с уточнением
TRESHOLD_YES_NO = 0.8       # порог отсечки при положительном ответе на уточнение
TRESHOLD_USER_QUIT = 0.8    # порог отсечки при проверке на отмену ввода параметров при режиме query


# скомпилим регулярочки
# <mailto:vndanilchenko@gmail.com|vndanilchenko@gmail.com>
pattern_mailto = re.compile(r'\<{1}[a-zA-Z0-9\:\@\.]{1,}\|{1}')
pattern_id = re.compile(r'^(\<{1}\@{1}[a-zA-Z0-9]{1,}\>{1})')
# <#CMX9X7F37|chat-bot-in-education>
pattern_channel1 = re.compile(r'\<{1}\#{1}[A-Z0-9]{1,}\|{1}')
pattern_rbracket = re.compile(r'\>')
pattern_email = re.compile(r'[0-9a-zа-я_]{1,}\@{1}[a-z]{1,}\.{1}[a-z]{1,}')

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
        self.commands = ['send_info', 'show_weather', 'get_schedule', 'tell_joke', 'send_email']
        self.classifier = models.Classifier()
        self.state_in = {'intent': None, 'command': None, 'state': 'normal', 'reply': None, 'query_params': {}}
        self.state_out = {'intent': None, 'command': None, 'state': 'normal', 'reply': None, 'query_params': {}}
        self.send_email = email.Send_email()
        self.weather_descr = pd.read_csv('./data/Multilingual_Weather_Conditions.csv', sep=',', encoding='utf-8')

    def set_default_state(self):
        """
        :return:
        """
        self.state_out = {'intent': self.state_in['intent'], 'command': self.state_in['command'], 'state': 'normal', 'reply': self.state_in['reply'], 'query_params': {}}

    # функция ищет среди каналов заданное название и возвращает id, либо None
    def get_channel_id(self, channel_name):
        req = self.webclient.conversations_list(types="public_channel, private_channel")
        for chanel in req.data['channels']:
            # print(chanel)
            if chanel['name'] == channel_name:
                return chanel['id']
        else:
            return None

    # основная функция
    def detect_message(self, **kwargs):
        """
        :param kwargs:
        :return:
        """
        # try:
        #     print('kwargs: ', kwargs)
        data = kwargs['data']
        # print('kwargs: ', data)
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
        # если вдруг начнет крэшиться просто пропустить
        # except:
        #     pass

    # функция проверяет заполненность параметров
    def get_query_params(self):
        """
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
                        self.state_out = self.state_in
                        return f'параметр не распознан:\n {params[param]}'
                    else:
                        self.state_in['query_params']['channel_id'] = channel
                        self.state_in['query_params'][param] = self.text_in
                elif param == 'ok':
                    max_score, max_class = self.classifier.predict_proba(self.text_in, model_flag=1)
                    print('ПРОВЕРКА ДА/НЕТ: max_score: ', max_score, 'max_class', max_class)
                    if max_class == 1 and max_score > TRESHOLD_YES_NO:
                        self.state_in['query_params']['ok'] = True
                        print('прошли проверку на наличие подтвреждения на отправку - положительно')
                    elif max_class == 0 and max_score > TRESHOLD_YES_NO:
                        # выходим из режима при уверенном отказе
                        print('прошли проверку на наличие подтвреждения на отправку - негативно')
                        return self.realize_user_quit()
                else:
                    self.state_in['query_params'][param] = self.text_in
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
                print(f"уточняем намерените {self.state_in['intent']}")
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


    # принимает метку класса и возвращает название интента
    def return_intent(self, max_class):
        """
        :param max_class:
        :return:
        """
        intent_line = self.responses_df[self.responses_df['class']==str(max_class)]
        intent = intent_line.intent.values[0]
        command = intent_line.command.values[0]
        print('команда до преобразования: ', command)
        command = None if command not in self.commands else command
        print('команда после преобразования: ', command)
        reply = intent_line.reply_now.values[0]
        print('прошел intent ')
        return intent, command, reply

    # преобразует строковый формат даты и времени из разных полей в одну переменную datetime
    def convert_datetime(self, temp_date, temp_time):
        """
        :param temp_date:
        :param temp_time:
        :return:
        """
        # return (datetime.strptime(str(temp_date), '%Y-%m-%d %H:%M:%S') + (datetime.combine(date.min, temp_time) - datetime.min))
        return (datetime.strptime(str(temp_date), '%m/%d/%Y') + (datetime.combine(date.min, datetime.strptime(str(temp_time), '%H:%M').time()) - datetime.min))

    # выполняет определенную команду, если такая присутствует в интенте
    def do_command(self, command, reply):
        """
        :param command:
        :param reply:
        :return:
        """
        response = reply
        # проверка на наличие команды
        if command == 'get_schedule':
            print('зашел в команду')
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

            # temp_df = None
        elif command == 'tell_joke':
            url = 'http://rzhunemogu.ru/RandJSON.aspx?CType=1'
            try:
                result = json.loads(requests.get(url).text, strict=False)
                response = result['content']
            except:
                print(self.catch_problems(self), ' команда:', command)
                self.do_command(self, 'tell_joke', response)
            print(f'ответ по команде: {response}')
        elif command == 'show_weather':
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
        elif command == 'send_email' or command == 'send_info':
            print('зашли в отправку письма')
            # response = self.send_email.run(addr_to="vndanilchenko@gmail.com", theme="тестовое сообщение", body="привет, Вадим! Это тестовое сообщение от бота Нетологии")
            self.state_in = self.state_out
            self.state_in['state'] = 'query'
            response = self.get_query_params()
            print(f'ответ по команде: {response}')
        else:
            print('НЕОПИСАННАЯ КОМАНДА!')
            response = 'без команды зашел: {}'.format(self.commands)
        return response

    # если что-то пошло не так
    def catch_problems(self):
        """
        :return:
        """
        self.set_default_state()
        return 'что-то пошло не так..'

    # если пользователь отказался подтверждать выполнение скрипта
    def realize_user_quit(self):
        """
        :return:
        """
        self.set_default_state()
        return 'хорошо, команда отменена.\nвсегда можете попробовать сначала ;)'

    # возвращает стандартное сообщение, если скор не проходит трэшолд
    def terminal_response(self):
        """
        :return:
        """
        self.set_default_state()
        return f"<@{self.user}>, \nя не понял, попробуй перефразировать"

    # возвращает стандартное сообщение, если ответ клиента НЕТ
    def negative_response(self):
        """
        :return:
        """
        self.set_default_state()
        return f"<@{self.user}>, \nвозможно я еще смогу тебе помочь, попробуй иначе поставить вопрос"

    # функция проверяет запрос на прохождение отсечек и режим
    def make_response(self):
        """
        :return:
        """
        # model = check_kb.recognize_kb()
        # model.predict_proba('привет как жизнь чем занимаешься работаешь или учишься', True)
        if self.state_in['state'] == 'approval':
            print('APPROVAL')
            # # проверка классификаторов ДА/НЕТ и отсечку по трэшолду
            max_score, max_class = self.classifier.predict_proba(self.text_in, model_flag=1)
            if max_class==1 and max_score > TRESHOLD_YES_NO:
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
            if max_class==1 and max_score > TRESHOLD_USER_QUIT:
                # выходим из режима
                reply = self.realize_user_quit()
            else:
                # функция обработает входной интент и проверит на заполнение всех атрибутов
                print('получили признак query, перенаправили сообщение в функцию')
                reply = self.get_query_params()

        else:
            # prediction_kb = self.recognize_kb.kb_predict_proba(self.text_in)
            max_score, max_class = self.classifier.predict_proba(self.text_in, model_flag=0)
            print(f"база распознана {max_score}" if max_score>TRESHOLD_KB else f"запрос не в базе знаний {max_score}")
            if max_score > TRESHOLD_KB: # Был prediction_kb
                # return f"Hi <@{self.user}>! \n I know what you mean.."
                # max_score, max_class = self.recognize_kb.intent_predict_proba(self.text_in)
                print('прошел СNN')
                intent, command, reply = self.return_intent(max_class)
                self.state_out = {'intent': intent, 'command': command, 'state': 'normal', 'reply': reply, 'query_params': {}}
                print('нашел интент {}, команду {}, скор {}'.format(intent, command, max_score))
                if max_score > TRESHOLD_EXPLICIT:
                    print('EXPLICIT')
                    print(f'прошел отсечку по скору')
                    if command:
                        print('есть ответ по команде: {}'.format(command))
                        reply = self.do_command(command, reply)
                    # return reply
                elif max_score > TRESHOLD_APPROVAL:
                    self.state_out = {'intent': intent, 'command': command, 'state': 'approval', 'reply': reply, 'query_params': {}}
                    reply = 'ты хочешь {}?'.format(intent)

                else:
                    reply = self.terminal_response()
                    self.set_default_state()
            else:
                reply = self.terminal_response()
                self.set_default_state()
        return reply

    def reply(self):
        """
        :return:
        """
        for channel in self.channels:
            self.web_client.chat_postMessage(
                channel=channel
                ,text=self.text_out # + " " + str(self.direct_link)# str(self.channels)
                # thread_ts=self.thread_ts
                ,as_user='UN1S55TNY'
            )

    def run(self):
        """
        :return:
        """
        print('{:+^50}'.format(' START '))
        self.rtmclient.on(event='message', callback=self.detect_message)
        self.rtmclient.start()


if __name__ == '__main__':
    bot = Slackbot()
    bot.run()
