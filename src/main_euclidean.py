'''
	Clase de prueba de los algoritmos de clustering
'''

from sklearn.cluster import KMeans, AgglomerativeClustering
import nltk
from nltk.cluster import KMeansClusterer
from readDataTweet89 import *
from readDataReuters import *
from metrics import homogenity, completeness, NMI
import preprocessing as pre
import csv
from sklearn import preprocessing


if __name__ == "__main__":

	#Leemos los embeddings 
	#COMENTAR CUANDO HAGAMOS TF-IDF
	embeddings, vocabulary = pre.read_embeddings("../crawl-300d-2M.vec")


	#Leemos los datos

	#Embeddings-concatenando
	#data = ReaderTweet89("../data/Tweet", "embeddings", embeddings, vocabulary, True)

	#Embeddings-media
	#data = ReaderTweet89("../data/20ng.txt", "embeddings", embeddings, vocabulary)
	data = ReaderReutersR52("../data/r52-train-all-terms.txt", "../data/r52-test-all-terms.txt", "embeddings", embeddings, vocabulary)

	#tf-idf
	#data = ReaderTweet89("../data/20ng.txt", "tfidf")
	#data = ReaderReutersR52("../data/r52-train-stemmed.txt", "../data/r52-test-stemmed.txt", "tfidf")
	
	tweets = data.get_vectors()
	labels_true = data.get_clusters()

	#tweets = preprocessing.normalize(tweets)

	print("Etiquetas reales: ", labels_true)

	#Aplicamos kmeans
	kmeans = KMeans(n_clusters = 20, random_state = 1234567, n_init = 2, max_iter = 100).fit(tweets)

	#kclusterer = KMeansClusterer(20, distance=nltk.cluster.util.cosine_distance, repeats=10)
	#assigned_clusters = kclusterer.cluster(tweets, assign_clusters=True)

	#agglomerative = AgglomerativeClustering(n_clusters = 20, affinity = "cosine", linkage = "complete").fit(tweets)

	print("Etiquetas predichas: ", kmeans.labels_)
	
	nmi = NMI(labels_true, kmeans.labels_)
	print(nmi)

