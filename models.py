"""
# module predicts intents
# @author: vndanilchenko@gmail.com
"""
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import re, joblib
from pymystem3 import Mystem
from difflib import ndiff
from tqdm import trange
import warnings
warnings.filterwarnings('ignore')

LEVENSTEIN_DIST = 2
THRESHOLD_FUTURE = 0.9

class Classifier:

    def __init__(self):
        self.bazonka = joblib.load('./data/kb.pkl')
        self.wvec = Word2Vec.load('./models/word2vec.model')
        self.word_list = joblib.load('./data/word_list.pkl')
        self.vectors = joblib.load('./data/kb_vectors.pkl')
        self.vectors_norm = np.vstack([np.dot(self.vectors[i], self.vectors[i]) for i in range(self.vectors.shape[0])]).squeeze()
        self.classifier = joblib.load('./models/clf.pkl')
        self.morph = Mystem()
        self.is_training = False
        self.is_past = False
        self.is_future = False
        print('инициализация Classifier')


    def levenshtein_distance(self, word):
        """
        расстояние Левенштейна между входящим запросом и известными моделям словами
        :param word:
        :return:
        """
        most_similar_words = {}
        for i in range(len(self.word_list)):
            counter = {"+": 0, "-": 0}
            distance = 0
            temp_word = self.word_list[i]
            for edit_code, *_ in ndiff(word, temp_word):
                if edit_code == " ":
                    distance += max(counter.values())
                    counter = {"+": 0, "-": 0}
                else:
                    counter[edit_code] += 1
            distance += max(counter.values())
            # сразу возвращаем, если все находим слово с отличием в одну букву
            if distance == 1:
                return temp_word
            elif distance <= LEVENSTEIN_DIST:
                most_similar_words[temp_word] = distance
        if most_similar_words:
            return list(most_similar_words.keys())[0]
        else:
            return None

    def check_phrase(self, phrase):
        """
        проверяет слова входящей фразы на наличие в словаре, в случае чего пробует найти ближайшее на заданной расстоянии
        :param phrase:
        :return:
        """
        new_phrase = []
        phrase_checked = []
        for word in phrase.split(' '):
            # если слова нет в словаре, пробуем найти похожее по Левенштейну
            if word not in self.word_list:
                word_lv = self.levenshtein_distance(word)
                if word_lv:
                    word = word_lv
            phrase_checked.append(word)
            new_phrase.append(word)
        return ' '.join(new_phrase)

    def preprocess(self, text):
        """
        предобработка и нормализация фразы
        :param text:
        :return:
        """
        text = str(text).lower().strip()
        text = re.sub(r'[^a-zа-я]', ' ', text)
        text = re.sub(r'([a-zа-яёе])\1{2,}', r'\1\1', text)  # aaaaa -> aa
        text = re.sub(r'([^a-zа-яёе0-9])\1{1,}', r'\1', text).strip()  # )))) -> )
        if not self.is_training:
            text = self.check_phrase(text)
            print('\nпреобразованная фраза: {}'.format(text))
            text = [word for word in self.morph.lemmatize(text) if not any(i in word for i in [' ', '\n'])]
            print('нормализованная фраза: {}'.format(' '.join(text)))
        return text

    def vectorize(self, phrase):
        """
        принимает на вход токенизированную фразу, возвращает усредненный вектор
        :param phrase:
        :return:
        """
        temp_vector = []
        for word in phrase:
            if word in self.wvec:
                temp_vector.append(self.wvec.wv.__getitem__(word))
            else:
                temp_vector.append(np.zeros(self.wvec.wv.vector_size))
        return np.mean(temp_vector, axis=0)


    def cosine_scores(self, q):
        """
        считает косинусное расстояние между запросом и фразами базы знаний, возвращает макс скор и индекс строки
        1 - вектора идентичны, 0 - ортогональны
        :param vectors: матриц векторов фраз из базы знаний [NxM]
        :param vectors_norm: произведений матрицы самой на себя (вектор норм)
        :param q: вектор запроса
        :param qn: произведение вектора запроса самого на себя (скаляр - норма)
        :return: вектор расстояний [Nx1]
        """
        qn = np.dot(q, q)
        sims = np.dot(self.vectors, q)
        dst = np.dot(self.vectors_norm, qn)
        dst[dst == 0] = np.inf
        similarity = sims * np.sqrt(1/dst)
        return max(similarity), np.argmax(similarity)

    def predict_proba(self, phrase, model_flag=0):
        """
        основная функция - предобрабатывает фразу, векторизует и запускает модель в зависимости от типа запроса
        model_flag==0 - возвращает скор наиболее близкой фразы к запросу по базе знаний и ее класс
        model_flag==1 - классификатор согласия (yesno)
        model_flag==2  - классификатор выхода из сценария (user_exit)
        model_flag==3 - проверка будущего времени (future)
        :param phrase:
        :return:
        """
        predictions = (0.0, 0)
        try:
            tokens = self.preprocess(phrase)
            vector = self.vectorize(tokens)
            if model_flag==0:
                max_score, max_row = self.cosine_scores(vector)
                print('ближайшая фраза: ', self.bazonka[self.bazonka.index == max_row]['message'].values[0])
                max_class = self.bazonka[self.bazonka.index == max_row]['TARGET'].values[0]
                predictions = (max_score, max_class)
                # TODO: сделать этот костыль поэлегантнее
            elif any(token in ['да', 'уверен', 'уверена', 'уверенный', 'нет', 'отмена', 'отбой', 'стоп', 'остановить', 'остановись'] for token in tokens):
                if model_flag==1:
                    if any(token in ['да', 'уверен', 'уверена', 'уверенный', 'ага'] for token in tokens):
                        predictions = (1.0, 1)
                    elif any(token in ['нет', 'отмена', 'отбой', 'стоп', 'остановить', 'остановись'] for token in tokens):
                        predictions = (1.0, 0)
                elif model_flag==2:
                    if any(token in ['отмена', 'отбой', 'стоп', 'остановить', 'остановись'] for token in tokens):
                        predictions = (1.0, 1)
            else:
                predictions = self.classifier[model_flag]['clf'].predict_proba(vector.reshape(1, -1))[0]
                predictions = (max(predictions), np.argmax(predictions))
            return predictions
        except Exception as e:
            print('во время предсказания возникла ошибка: ', e.args)
            return predictions

    def set_default_time_states(self):
        """
        обнуляет состояния времени
        :return:
        """
        self.is_future = False
        self.is_past = False

    def check_time(self, text):
        """
        функция проверяет на наличие временных меток во входящей фразе, от нее будет зависеть поле ответа
        :param text: исходный запрос, text-in
        :return:
        """
        self.set_default_time_states()
        # сначала сделаем метки на время, потом сравним и определим приоритет при одновременном наступлении
        if any(word in ['вчера', 'раньше', 'ранее', 'прошлое', 'прошлом', 'до', 'было', 'был'] for word in str(text).split(' ')):
            self.is_past = True
        else:
            for word in self.morph.analyze('отправить email'):
                if 'analysis' in word:
                    if word['analysis']:
                        if 'пе=прош' in word['analysis'][0]['gr']:
                            self.is_past = True
                            break

        pred_f = self.predict_proba(text, model_flag=3)
        if pred_f[0] > THRESHOLD_FUTURE and pred_f[1] == 1:
            self.is_future = True
        elif any(word in ['завтра', 'дальше', 'будущее', 'потом', 'будешь', 'далее', 'через', 'будет', 'следующий', 'следующем', 'следующих'] for word in str(text).split(' ')):
            self.is_future = True

        # примем решение что возвращать
        if all([self.is_future, self.is_past]):
            return 'future'
        elif not any([self.is_future, self.is_past]):
            return 'normal'
        elif self.is_future:
            return 'future'
        elif self.is_past:
            return 'past'
        else:
            return 'error'

    def fit(self):
        """
        модуль при запуске пересоздает словари, преобцчает модель word2vec и сохраняет новые вектора фраз базы знаний
        :return:
        """
        # True: не будет запущен Левенштейн по старому словарю

        self.is_training = True
        try:
            bazonka = pd.read_excel('./kb/kb.xlsx', sheet_name='kb')['message'].tolist()
            data_exit = pd.read_excel('./kb/data_train_user_exit.xlsx')['message'].tolist()
            data_yesno = pd.read_excel('./kb/data_train_yesno.xlsx')['message'].tolist()
            data_future = pd.read_excel('./kb/data_train_future.xlsx')['message'].tolist()
            concatted_phrases = bazonka + data_exit + data_yesno + data_future

            # сохраним базу знаний
            joblib.dump(bazonka, './data/kb.pkl')

            # создаем словарь исходных слов по обучающим выборкам
            word_list = set()
            print('--- создаем словарь исходных слов по обучающим выборкам ---')
            for sentence_id in trange(len(concatted_phrases)):
                sequence = self.preprocess(concatted_phrases[sentence_id])
                for word in sequence:
                    if word not in word_list:
                        word_list.add(word)
            # сохраним словарь
            joblib.dump(word_list, './data/kb_vectors.pkl')

            # возвращаем флаг в исходное состяние, чтобы начать делать проверку и нормализовать слова
            self.is_training = False

            # создаем последовательность нормализованных фраз, разбитых на токены
            print('--- создаем последовательность нормализованных фраз, разбитых на токены ---')
            sequences = []
            for sentence_id in trange(self.bazonka.shape[0]):
                sequences.append(self.preprocess(self.bazonka.message.iloc[sentence_id]))

            # обучим модель word2vec
            print('--- обучаем word2vec модель ---')
            wvec = Word2Vec(sentences=sequences, size=50, window=5, min_count=1, seed=777, workers=-1, max_vocab_size=None)
            wvec.save('./models/word2vec.model')

            # обновим вектора базы знаний
            print('--- обновляем вектора базы знаний ---')
            vectors = []
            for i in trange(len(sequences)):
                vectors.append(self.vectorize(sequences[i]))
            self.vectors = np.asarray(vectors)
            joblib.dump(self.vectors, './data/kb_vectors.pkl')
            print('{:+^150}'.format('done'))
        except Exception as e:
            print('при обучении возникла ошибка: ', e.args)
            self.is_training = False



if __name__ == '__main__':
    object = Classifier()
    object.check_time('было')
    # object.fit()
    # object.preprocess('да')
    object.vectorize(object.preprocess('привет'))
    # score, max_class = object.predict_proba('что ты умеешь', model_flag=0)
    # object.morph.analyze('будущем')
