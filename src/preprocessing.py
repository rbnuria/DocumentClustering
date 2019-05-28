'''
	Fichero para preprocesamiento de texto.
'''

from nltk import word_tokenize
import io
import numpy as np

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


def word2embeddings(data, embedding_path):
	fin = io.open(embedding_path, 'r', encoding='utf-8', newline='\n', errors='ignore')

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

	print("Sustituyendo palabras por su embedding correspondiente ...")
	data_embeddings = []
	for sentence in data:
		sentence_embedding = []
		
		for word in sentence:
			if word == 0:
				sentence_embedding.append(np.array(embeddings_matrix[vocabulary['PADDING']]))
			elif word in vocabulary:
				sentence_embedding.append(np.array(embeddings_matrix[vocabulary[word]]))
			else:
				sentence_embedding.append(np.array(embeddings_matrix[vocabulary['UNKOWN']]))

		sentence_embedding = np.array(sentence_embedding)
		vector_medias = sentence_embedding.mean(1)
		vector_medias = np.array(vector_medias)

		data_embeddings.append(vector_medias)

	data_embeddings = np.array(data_embeddings)
	print(data_embeddings.shape)

	return data_embeddings
