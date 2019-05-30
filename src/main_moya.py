'''
	Clase de prueba de los algoritmos de clustering
'''

from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from readDataTweet89 import *
from readDataReuters import *
from metrics import homogenity, completeness, NMI
import preprocessing
import time
from scipy.cluster.hierarchy import dendrogram, linkage




if __name__ == "__main__":

	#Leemos los embeddings 
	#COMENTAR CUANDO HAGAMOS TF-IDF
	seeds = [112, 1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 123456789]
	epsilons = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,0.55, 0.6, 0.65 ]
	for i in range(10):
		t1 = time.time()

		#embeddings, vocabulary = preprocessing.read_embeddings("../crawl-300d-2M.vec")

		#Leemos los datos

		#Embeddings-concatenando
		#data = ReaderTweet89("../data/20ng.txt", "embeddings", embeddings, vocabulary, True)

		#Embeddings-media
		#data = ReaderTweet89("../data/20ng.txt", "embeddings", embeddings, vocabulary)
		d#ata = ReaderReutersR52("../data/r52-train-all-terms.txt", "../data/r52-test-all-terms.txt", "embeddings", embeddings=embeddings, vocab = vocabulary)

		#tf-idf
		#data = ReaderTweet89("../data/20ng.txt", "tfidf")
		data = ReaderReutersR52("../data/r52-train-all-terms.txt", "../data/r52-test-all-terms.txt","tfidf")


		tweets = data.get_vectors()
		labels_true = data.get_clusters()
		print(tweets.shape)
		print("Etiquetas reales: ", labels_true)

		t2 = time.time()

		t3 = time.time()

		#kmeans
		#1234567
		kmeans = KMeans(n_clusters = 52, random_state = seeds[i], n_init = 10, max_iter = 300).fit(tweets)
		
		#AGL-EUC-WARD
		#AGL_EUC1 = AgglomerativeClustering(n_clusters=20, affinity="euclidean", linkage="ward").fit(tweets)
		#AGL_EUC1 = linkage(tweets)
		#dendrogram(AGL_EUC1)
		#DBSCAN

		#birch = Birch(threshold = epsilons[i], n_clusters = 20).fit(tweets)

		print("Etiquetas predichas K: ", kmeans.labels_)
		nmi = NMI(labels_true, kmeans.labels_)
		print("nmi: ", nmi)
		t4 = time.time()
		print("tiempo 1: ", t2-t1)
		print("tiempo 2: ", t4-t3)
		

