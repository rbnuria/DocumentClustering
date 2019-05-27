#Fichero para la lectura de datos

from nltk.corpus import reuters
import os
from docx import Document
from bs4 import BeautifulSoup



def collection_stats():
	#Lista de documentos
	documents = reuters.fileids()
	print("Read " + str(len(documents)) + " documents")

	train_docs = list(filter(lambda doc: doc.startswith("train"), documents))
	train_docs = list(filter(lambda doc: doc.startswith("train"),
                        documents));
	print(str(len(train_docs)) + " total train documents")

	test_docs = list(filter(lambda doc: doc.startswith("test"), documents))
	print(str(len(test_docs)) + " total test documents")

	# List of categories
	categories = reuters.categories()
	print(str(len(categories)) + " categories")

	# Documents in a category
	category_docs = reuters.fileids("acq")

	# Words for a document
	document_id = category_docs[0]
	document_words = reuters.words(category_docs[0])
	print(document_words)

	# Raw document
	print(reuters.raw(document_id))

	return documents


if __name__ == "__main__":
	collection_stats()




