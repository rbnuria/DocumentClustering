'''
	Fichero para preprocesamiento de texto.
'''

from nltk import word_tokenize
from nltk.stem import PorterStemmer
import io
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


def tokenize(text):
	return word_tokenize(text)

def padding_truncate(sentences, max_length):

	print("Padding truncate...")
	for i in range(len(sentences)):
		sent_size = len(sentences[i])

		if sent_size > max_length:
			sentences[i] = sentences[i][:max_length]
		elif sent_size < max_length:
			if(isinstance(sentences[i],list)):
				sentences[i] += [0] * (max_length - sent_size)
			else:
				list_sentence = sentences[i].tolist()
				list_sentence += [0] * (max_length - sent_size)
				sentences[i] = np.array(list_sentence)

	return sentences


def word2tfidf(data):
	#Inicializamos tfidf
	vectorizer = TfidfVectorizer()

	X = vectorizer.fit_transform(data)

	print("Vectorizer")
	print(X.shape)

	return X

def apply_stemmer_stopword(data):
	ps = PorterStemmer()

	stemmed_data = []

	for sentence in data:
		sentence_tokenized = delete_stopwords_sentence(word_tokenize(sentence))
		new_sentence = ""
		for word in sentence_tokenized:
			new_sentence += " " + ps.stem(word)

		new_sentence += "\n"

	stemmed_data.append(new_sentence)


def delete_stopwords(tokenized_data):
	stop_words = stopwords.words('english')

	new_data = []

	for sentence in tokenized_data:
		new_sentence = [word for word in sentence if word not in stop_words]
		new_data.append(new_sentence)

	return np.array(new_data)

def delete_stopwords_sentence(tokenized_sentence):
	stop_words = stopwords.words('english')

	new_sentence = [word for word in tokenized_sentence if word not in stop_words]

	return new_sentence

def read_embeddings(path_embeddings):
	fin = io.open(path_embeddings, 'r', encoding='utf-8', newline='\n', errors='ignore')

	#Hacemos vocabulario
	vocabulary = {}
	vocabulary["PADDING"] = len(vocabulary)
	vocabulary["UNKOWN"] = len(vocabulary)

	embeddings_matrix = []
	embeddings_matrix.append(np.zeros(300))
	embeddings_matrix.append(np.random.uniform(-0.25, 0.25, 300))


	print("Leyendo embeddings...")
	for line in fin:
		tokens = line.rstrip().split(' ')

		if(len(tokens[1:]) == 300):
			vocabulary[tokens[0]] = len(vocabulary)
			embeddings_matrix.append(np.array(tokens[1:]))

	return (embeddings_matrix, vocabulary)

def word2embeddings(data, embedding, vocab, concatenate):
	
	embeddings_matrix = embedding
	vocabulary = vocab

	print("Sustituyendo palabras por su embedding correspondiente ...")
	data_embeddings = []
	for sentence in data:
		sentence_embedding = []
		
		if concatenate:
			for word in sentence:
				if word == 0:
					sentence_embedding.extend(np.array(embeddings_matrix[vocabulary['PADDING']]).astype(np.float))
				elif word in vocabulary:
					sentence_embedding.extend(np.array(embeddings_matrix[vocabulary[word]]).astype(np.float))
				else:
					sentence_embedding.extend(np.array(embeddings_matrix[vocabulary['UNKOWN']]).astype(np.float))

			sentence_embedding = np.array(sentence_embedding)
			data_embeddings.append(sentence_embedding)

		else:
			for word in sentence:
				if word == 0:
					pass
					#sentence_embedding.append(np.array(embeddings_matrix[vocabulary['PADDING']]).astype(np.float))
				elif word in vocabulary:
					sentence_embedding.append(np.array(embeddings_matrix[vocabulary[word]]).astype(np.float))
				else:
					#sentence_embedding.append(np.array(embeddings_matrix[vocabulary['UNKOWN']]).astype(np.float))
					pass

			sentence_embedding = np.array(sentence_embedding)
			vector_medias = sentence_embedding.mean(0)
			vector_medias = np.array(vector_medias)

			data_embeddings.append(vector_medias)


	data_embeddings = np.array(data_embeddings)
	print(data_embeddings.shape)

	return data_embeddings
