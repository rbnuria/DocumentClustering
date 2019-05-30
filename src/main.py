'''
	Clase de prueba de los algoritmos de clustering
'''

from sklearn.cluster import KMeans
from readDataTweet89 import *
from readDataReuters import *
from metrics import homogenity, completeness, NMI
import preprocessing

if __name__ == "__main__":

	#Leemos los embeddings 
	#COMENTAR CUANDO HAGAMOS TF-IDF
	#embeddings, vocabulary = preprocessing.read_embeddings("../crawl-300d-2M.vec")

	#Leemos los datos

	#Embeddings-concatenando
	#data = ReaderTweet89("../data/Tweet", "embeddings", embeddings, vocabulary, True)

	#Embeddings-media
	#data = ReaderTweet89("../data/20ng.txt", "embeddings", embeddings, vocabulary)
	#data = ReaderReutersR52("../data/r52-train-stemmed.txt", "../data/r52-test-stemmed.txt", "embeddings", embeddings, vocabulary)

	#tf-idf
	data = ReaderTweet89("../data/20ng.txt", "tfidf")
	
	tweets = data.get_vectors()
	labels_true = data.get_clusters()

	print("Etiquetas reales: ", labels_true)

	#Aplicamos kmeans
	kmeans = KMeans(n_clusters = 20, random_state = 1234567, n_init = 10, max_iter = 100).fit(tweets)

	print("Etiquetas predichas: ", kmeans.labels_)
	
	nmi = NMI(labels_true, kmeans.labels_)
	print(nmi)

