import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import lightfm as lf
import nmslib
import pickle
import scipy.sparse as sparse
import plotly.express as px


@st.cache_data
def read_files(folder_name='data'):
    """
    Функция для чтения файлов
    Возвращает 2 DataFrame с рейтингами и характеристиками книг
    """
    ratings = pd.read_csv(folder_name + '/ratings.csv')
    books = pd.read_csv(folder_name + '/books.csv')
    
    return ratings, books

def make_mappers(books):
    """функция для отображения id в title и authors
    Возвращает 2 словаря:
      * Ключи первого словаря - идентификаторы книг, значения - их названия
      * Ключи второго словаря - идентификаторы книг, значения - их авторы
    """
    name_mapper = dict(zip(books.book_id, books.title))
    authors_mapper = dict(zip(books.book_id, books.authors))
    
    return name_mapper, authors_mapper

def load_embeddings(file_name='item_embeddings.pkl'):
    """Функция для загрузки векторных представлений
    Возвращает прочитанные ембеддинги книг и индекс(граф) для поиска похожих книг
    """
    with open(file_name, 'rb') as file:
        item_embeddings = pickle.load(file)
        
    # Тут мы используем nmslib, чтобы создать быстрый knn
    nms_ids = nmslib.init(method='hnsw', space='cosinesimil')
    nms_ids.addDataPointBatch(item_embeddings)
    nms_ids.createIndex(print_progress=True)
    
    return item_embeddings, nms_ids

def nearest_books_nms(book_id, index, n=10):
    """Функция для поиска ближайших соседей, возвращает построенный индекс
    Возвращает n наиболее похожих книг и расстояние до них
    """
    nn = index.knnQuery(item_embeddings[book_id], k=n)
    
    return nn

def get_recomendation_df(ids, distance, name_mapper, author_mapper):
    """Функция для составления таблицы из рекомендованных книг
    Возвращает DataFrame со столбцами:
     * book_name - название книги
     * book_author - автор книги
     * distance - значение метрики расстояния до книги
    """
    names = []
    authors = []
    #Для каждого индекса книги находим её название и автора
    #Результаты добавляем в списки
    for idx in ids:
        names.append(name_mapper[idx])
        authors.append(author_mapper[idx])
    
    #Составляем DataFrame
    recomendation_df = pd.DataFrame({
        'book_name': names,
        'book_author': authors,
        'distance': distance
    })
    
    return recomendation_df

#Загружаем данные
ratings, books = read_files()
#Создаём словари для сопоставления id книг и их названий/авторов
name_mapper, author_mapper = make_mappers(books)
#Загружаем эмбеддинги и создаём индекс для поиска
item_embeddings, nms_idx = load_embeddings()

st.title('Система рекомендации книг')

st.markdown("""Welcome to the web page of the book recommendation app!
This application is a prototype of a recommendation system based on a machine learning model.

To use the application, you need:
1. Enter the approximate name of the book you like
2. Select its exact name in the pop-up list of books
3. Specify the number of books you need to recommend

After that, the application will give you a list of books most similar to the book you specified""")

# Вводим строку для поиска книг
title = st.text_input('Введите название книги:', '')
title = title.lower()  
option = None
#Выполняем поиск по книгам — ищем неполные совпадения
if title:
    output = books[books['title'].apply(lambda x: x.lower().find(title)) >= 0]
    option = st.selectbox('Выберите нужную книгу:', output['title'].values)

if option:
    st.markdown(f'Вы выбрали: {option}')
    book_id = output[output['title'].values == option]['book_id'].values
print(option)
if option:
    count_recomendation = st.number_input(
        label='Укажите количество рекомендованных книг',
        value=10
    )
    ids, distance = nearest_books_nms(book_id, nms_idx, count_recomendation + 1)
    ids, distance = ids[1:], distance[1:]
    st.markdown('Наиболее подходящие книги:')
    df = get_recomendation_df(ids, distance, name_mapper, author_mapper)
    st.dataframe(df[['book_name', 'book_author']])
    
    fig = px.bar(
        data_frame=df,
        x='book_name',
        y='distance',
        hover_data=['book_author'],
        title='Косинусное расстояние между векторами книг'
    )
    st.write(fig)
    