'''
	Clase que implementa la clase abstracta Reader  para el caso concreto
	de la base de datos Reuters-21578 R52 

'''

from readData import Reader
import json
import numpy as np
import preprocessing

class ReaderReutersR52(Reader):

	def __init__(self, path_train, path_test, length, embeddings):
		self._path_train = path_train
		self._path_test = path_test
		self._max_length = length
		self._embedding_path = embeddings
		self.read_data()
		self.prepare_data()

	def read_data(self):
		self._text = []
		self._cluster = []

		print("Read data...")

		with open(self._path_train, 'r') as file_:
			for line in file_:
				tokens = line.split("\t")

				self._text.append(tokens[1])
				self._cluster.append(tokens[0])

			self._text = np.array(self._text)
			self._cluster = np.array(self._cluster)

		with open(self._path_test, 'r') as file_:
			for line in file_:
				tokens = line.split("\t")

				self._text.append(tokens[1])
				self._cluster.append(tokens[0])

			self._text = np.array(self._text)
			self._cluster = np.array(self._cluster)

	def prepare_data(self):
		print("Prepare data...")
		self._data = []

		#Tokenizamos
		for sentence in self._text:
			self._data.append(preprocessing.tokenize(sentence))

		#Padding-truncate
		self._data = preprocessing.padding_truncate(self._data, self._max_length)

		#Pasamos a embeddings
		self._vectors = preprocessing.word2embeddings(self._data, self._embedding_path)


	def get_text(self):
		return self._text

	def get_clusters(self):
		return self._cluster

	def get_data(self):
		return self._data

	def get_vectors(self):
		return np.array(self._vectors)

if __name__ == "__main__":
	reader = ReaderReutersR52("../data/r52-train-stemmed.txt", "../data/r52-test-stemmed.txt", 50, "../crawl-300d-2M.vec")
	data = reader.get_data()
	vectors = reader.get_vectors()

	print(data[0])
	print(vectors[0])