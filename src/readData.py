'''	
	Clase abstracta para la lectura de los conjuntos de datos.
'''

from abc import abstractmethod

class Reader:

	@abstractmethod
	def read_data(self):
		pass		

	@abstractmethod
	def prepare_data(self):
		pass

	@abstractmethod
	def get_text(self):
		pass

	@abstractmethod
	def get_data(self):
		pass

	@abstractmethod
	def get_clusters(self):
		pass