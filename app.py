import streamlit as st
import numpy as np
import pandas as pd
import lightfm as lf
import nmslib
import pickle
import scipy.sparse as sparse
import zipfile

def main():

	def nearest_books_nms(book_id, index, n=10):
		"""Функция для поиска ближайших соседей, возвращает построенный индекс"""
		nn = index.knnQuery(item_embeddings[book_id], k=n)
		return nn

	def get_names(index):
		"""
		input - idx of books
		Функция для возвращения имени книг
		return - list of names
		"""
		names = []
		for idx in index:
			if idx>0:
				names.append('Book name:  {} '.format(name_mapper[idx]) + '  Book Author: {}'.format(author_mapper[idx]))
			else:
				names.append('')
		return names

	def read_files(folder_name=''):
		"""
		Функция для чтения файлов + преобразование к  нижнему регистру
		"""
		zr = zipfile.ZipFile(folder_name+'\ratings.zip', 'r')
		ratings = pd.read_csv(zr.open('ratings.csv'))
		zb = zipfile.ZipFile(folder_name+'\books.zip', 'r')
		books = pd.read_csv(zb.open('books.csv'))
		books['title'] = books.title.str.lower()
		return ratings, books 

	def make_mappers():
		"""
		Функция для создания отображения id в title
		"""
		name_mapper = dict(zip(books.book_id, books.title))
		author_mapper = dict(zip(books.book_id, books.authors))

		return name_mapper, author_mapper

	def load_embeddings():
		"""
		Функция для загрузки векторных представлений
		"""
		with open('item_embeddings.pickle', 'rb') as f:
			item_embeddings = pickle.load(f)

		# Тут мы используем nmslib, чтобы создать наш быстрый knn
		nms_idx = nmslib.init(method='hnsw', space='cosinesimil')
		nms_idx.addDataPointBatch(item_embeddings)
		nms_idx.createIndex(print_progress=True)
		return item_embeddings,nms_idx
	
	
	#Загружаем данные
	ratings, books  = read_files(folder_name='data') 
	name_mapper, author_mapper = make_mappers()
	item_embeddings,nms_idx = load_embeddings()

	# Создадим приветственную форму (нет, не helloworld). Просто попросим пользователя ввести название книги.

	#Форма для ввода текста
	title = st.text_input('Book Name', '')
	title = title.lower()

	#Наш поиск по книгам
	output = books[books.title.str.contains(title) > 0]

	#Выбор книги из списка
	option = st.selectbox('Which book?', output['title'].values)

	#Выводим книгу
	'You selected: ', option

	#Ищем рекомендации
	val_index = output[output['title'].values == option].book_id
	index = nearest_books_nms(val_index, nms_idx, 5)

	#Выводим рекомендации к ней
	'Most simmilar books are: '
	st.write('', get_names(index[0])[1:])


if __name__ == "__main__":
    main()
