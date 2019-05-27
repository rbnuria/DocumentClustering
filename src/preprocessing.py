'''
	Fichero para preprocesamiento de texto.

	1- Lowercase the original content
	2- Tokenize
	3- Steam (o no, ya veremos)
'''

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")

def tokenize(text):
	
	return word_tokenize(text)

