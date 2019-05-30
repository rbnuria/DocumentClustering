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
	data_tweet = ReaderTweet89("../data/Tweet", "../data/20ng-test-all-terms.txt","embeddings", embeddings, vocabulary)
	#data = ReaderReutersR52("../data/r52-train-all-terms.txt", "../data/r52-test-all-terms.txt", "embeddings", embeddings, vocabulary)

	#tf-idf
	#data = ReaderTweet89("../data/20ng.txt", "tfidf")
	#data = ReaderReutersR52("../data/r52-train-stemmed.txt", "../data/r52-test-stemmed.txt", "tfidf")
	
	tweets = data.get_vectors()
	labels_true = data.get_clusters()

	#tweets = preprocessing.normalize(tweets)

	print("Etiquetas reales: ", labels_true)

	#Aplicamos kmeans
	kmeans_1 = KMeans(n_clusters = 89, random_state = 1234567, n_init = 1, max_iter = 100).fit(tweets)
	kmeans_2 = KMeans(n_clusters = 89, random_state = 15431341, n_init = 1, max_iter = 100).fit(tweets)
	kmeans_3 = KMeans(n_clusters = 89, random_state = 6666, n_init = 1, max_iter = 100).fit(tweets)
	kmeans_4 = KMeans(n_clusters = 89, random_state = 123477567, n_init = 1, max_iter = 100).fit(tweets)
	kmeans_5 = KMeans(n_clusters = 89, random_state = 326745, n_init = 1, max_iter = 100).fit(tweets)
	kmeans_6 = KMeans(n_clusters = 89, random_state = 231252, n_init = 1, max_iter = 100).fit(tweets)
	kmeans_7 = KMeans(n_clusters = 89, random_state = 124677, n_init = 1, max_iter = 100).fit(tweets)
	kmeans_8 = KMeans(n_clusters = 89, random_state = 564322, n_init = 1, max_iter = 100).fit(tweets)
	kmeans_9 = KMeans(n_clusters = 89, random_state = 257234, n_init = 1, max_iter = 100).fit(tweets)
	kmeans_10 = KMeans(n_clusters = 89, random_state = 99942, n_init = 1, max_iter = 100).fit(tweets)

	print("Etiquetas predichas: ", kmeans.labels_)
	
	nmi_1 = NMI(labels_true, kmeans_1.labels_)
	nmi_2 = NMI(labels_true, kmeans_2.labels_)
	nmi_3 = NMI(labels_true, kmeans_3.labels_)
	nmi_4 = NMI(labels_true, kmeans_4.labels_)
	nmi_5 = NMI(labels_true, kmeans_5.labels_)
	nmi_6 = NMI(labels_true, kmeans_6.labels_)
	nmi_7 = NMI(labels_true, kmeans_7.labels_)
	nmi_8 = NMI(labels_true, kmeans_8.labels_)
	nmi_9 = NMI(labels_true, kmeans_9.labels_)
	nmi_10 = NMI(labels_true, kmeans_10.labels_)

	#print(nmi)

	

