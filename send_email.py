"""
# module helps to send info via email
# @author: vndanilchenko@gmail.com
"""

import smtplib  # Импортируем библиотеку по работе с SMTP

# Добавляем необходимые подклассы - MIME-типы
from email.mime.multipart import MIMEMultipart  # Многокомпонентный объект
from email.mime.text import MIMEText  # Текст/HTML
import os

# импортируем токены
try:
    pwd = os.environ.get('MAIL_PWD')
except:
    from private.email import pwd
try:
    email = os.environ.get('MAIL_ADDRESS')
except:
    from private.email import email_yandex as email



class Send_email:

    def __init__(self):
        self.addr_from = email  # Адресат
        self.addr_to = None
        self.password = pwd  # Пароль
        print('инициализация Send_email: ', email)

    def run(self, addr_to, theme, body):
        msg = MIMEMultipart()  # Создаем сообщение
        msg['From'] = self.addr_from  # Адресат
        msg['To'] = addr_to  # Получатель
        msg['Subject'] = theme #  # Тема сообщения
        msg.attach(MIMEText(body, 'plain'))  # Добавляем в сообщение текст

        server = smtplib.SMTP('smtp.yandex.com:587')  # Создаем объект SMTP
        # server.set_debuglevel(True)  # Включаем режим отладки - если отчет не нужен, строку можно закомментировать
        server.starttls()  # Начинаем шифрованный обмен по TLS
        server.login(user=self.addr_from, password=self.password)  # Получаем доступ
        server.send_message(msg)  # Отправляем сообщение
        server.quit()  # Выходим
        return 'сообщение на адрес "{}" отправлено'.format(addr_to)

# if __name__ == '__main__':
#     obj = Send_email()
#     obj.run(addr_to="@gmail.com", theme="тестовое сообщение", body="привет, друг! Это тестовое сообщение от бота Нетологии")