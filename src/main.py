'''
	Clase de prueba de los algoritmos de clustering
'''

from sklearn.cluster import KMeans, AgglomerativeClustering
from readDataTweet89 import *
from readDataReuters import *
from metrics import homogenity, completeness, NMI
import preprocessing as pre
import csv
from sklearn import preprocessing


if __name__ == "__main__":

	#Leemos los embeddings 
	#COMENTAR CUANDO HAGAMOS TF-IDF
	#embeddings, vocabulary = pre.read_embeddings("../crawl-300d-2M.vec")

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



	#f = open("tweets.csv", "wb")
	#writer = csv.writer(f)

	#writer.write(tweets)

	#f1 = open("labels.csv", "wb")
	#writer = csv.writer(f1)

	#writer.write(labels_true)

	tweets = preprocessing.normalize(tweets)

	print("Etiquetas reales: ", labels_true)

	#Aplicamos kmeans
	kmeans = KMeans(n_clusters = 20, random_state = 1234567, n_init = 10, max_iter = 100).fit(tweets)

	agglomerative = AgglomerativeClustering(n_clusters = 20, affinity = "cosine", linkage = "complete")

	print("Etiquetas predichas: ", agglomerative.labels_)
	
	nmi = NMI(labels_true, agglomerative.labels_)
	print(nmi)

