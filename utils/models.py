import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import string, re, joblib
from tqdm import trange
from pymystem3 import Mystem
from difflib import ndiff


LEVENSTEIN_DIST = 2

class Classifier:

    def __init__(self):
        self.bazonka = pd.read_excel('./data/kb_new.xlsx', sheet_name='kb')
        self.wvec = Word2Vec.load('./models/word2vec.model')
        # self.id2word = joblib.load('./data/id2word.pkl')
        self.word_dict = joblib.load('./data/dict.pkl')
        self.vectors = joblib.load('./data/kb_vectors.pkl')
        self.vectors_norm = np.vstack([np.dot(self.vectors[i], self.vectors[i]) for i in range(self.vectors.shape[0])]).squeeze()
        self.morph = Mystem()


    def levenshtein_distance(self, word):
        """
        :param word:
        :return:
        """
        most_similar_words = {}
        for i in range(len(self.word_dict)):
            counter = {"+": 0, "-": 0}
            distance = 0
            temp_word = self.word_dict[i]
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
        проверят слова входящей фразы на наличие в словаре, в случае чего пробует найти ближайшее на заданной расстоянии
        :param phrase:
        :return:
        """
        new_phrase = []
        phrase_checked = []
        for word in phrase.split(' '):
            # если слова нет в словаре, пробуем найти похожее по Левенштейну
            if word not in self.word_dict:
                word_lv = self.levenshtein_distance(word)
                if word_lv:
                    word = word_lv
            phrase_checked.append(word)
            new_phrase.append(word)
        print('преобразованная фраза: {}'.format(' '.join(phrase_checked)))
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
        text = self.check_phrase(text)
        return [word for word in self.morph.lemmatize(text) if not any(i in word for i in [' ', '\n'])]

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
        основная функция - предобрабатывает фразу, векторизует, возвращает наиболее близкую фразу к запросу по базе знаний
        :param phrase:
        :return:
        """
        tokens = self.preprocess(phrase)
        vector = self.vectorize(tokens)
        if model_flag==0:
            max_score, max_row = self.cosine_scores(vector)
            print('ближайшая фраза: ', self.bazonka[self.bazonka.index == max_row]['message'].values[0])
            max_class = self.bazonka[self.bazonka.index == max_row]['TARGET'].values[0]
            predictions = (max_score, max_class)
        # классификатор согласия
        elif model_flag==1:
            # predictions = self.model_yes_no(sequences)
            predictions = (0.99, 0)
        # классификатор выхода из сценария
        elif model_flag==2:
            predictions = (0.99, 0)
            # predictions = self.model_user_quit(sequences)
        return predictions


if __name__ == '__main__':

    # vectors = []
    # for i in trange(bazonka.shape[0]):
    #     vectors.append(vectorize(preprocess(bazonka.message.iloc[i])))
    # vectors
    # vectors = np.asarray(vectors)

    # joblib.dump(vectors, './data/kb_vectors.pkl')
    # vectors = joblib.load('./data/kb_vectors.pkl')
    object = Classifier()
    score, max_class = object.intent_predict_proba('метеапрогноз', model_flag=0)
    res = object.levenshtein_distance('метеапрогноз')