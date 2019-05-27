'''
	Clase de prueba de los algoritmos de clustering
'''

from sklearn.cluster import KMeans
from readDataTweet89 import *

if __name__ == "__main__":
	#Leemos los datos
	tweets = ReaderTweet89("../data/Tweet", 20, "../crawl-300d-2M.vec").get_vectors()

	print(tweets.shape)

	#Aplicamos kmeans
	kmeans = KMeans(n_clusters = 2, random_state = 1234567, n_init = 1, max_iter = 100).fit(tweets)

	print(kmeans.labels_)

