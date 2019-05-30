'''
	Clase que implementa la clase abstracta Reader  para el caso concreto
	de la base de datos Tweet89 y 20ng

'''

from readData import Reader
import json
import numpy as np
import preprocessing

class ReaderTweet89(Reader):

	def __init__(self, path, type_, embeddings = None, vocab = None, concatenate = False):
		self._path = path
		self._type = type_
		self._concatenate = concatenate
		self._embedding = embeddings
		self._vocabulary = vocab
		self.read_data()
		self.prepare_data()

	def read_data(self):
		self._text = []

		self._cluster = []

		print("Read data...")
		with open(self._path, 'r') as json_file:
			for line in json_file:
				data = json.loads(line)
				self._text.append(data['text'])
				self._cluster.append(data['cluster'])

			self._text = np.array(self._text)
			self._cluster = np.array(self._cluster)
			
	def prepare_data(self):
		print("Prepare data...")

		#Pasamos a embeddings
		if self._type == "embeddings":
			self._data = []
	
			for sentence in self._text:
				self._data.append(preprocessing.tokenize(sentence))

			if self._concatenate:
				max_length = 0
				for sentence in self._data:
					if len(sentence) > max_length:
						max_length = len(sentence)

				self._data = preprocessing.padding_truncate(self._data, max_length)

			print(self._data[1])
			print(self._data[2])

			self._data = preprocessing.delete_stopwords(self._data)
			self._vectors = np.array(preprocessing.word2embeddings(self._data, self._embedding, self._vocabulary, self._concatenate))
		else:
			self._text = preprocessing.apply_stemmer_stopword(self._text)
			self._vectors = preprocessing.word2tfidf(self._text)


	def get_text(self):
		return self._text

	def get_clusters(self):
		return self._cluster

	def get_vectors(self):
		return self._vectors
