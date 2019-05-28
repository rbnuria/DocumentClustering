from sklearn.metrics import normalized_mutual_info_score, homogeneity_score, completeness_score

def NMI(labels_true, labels_pred):
	return normalized_mutual_info_score(labels_true, labels_pred)

def homogenity(labels_true, labels_pred):
	return homogeneity_score(labels_true, labels_pred)

def completeness(labels_true, labels_pred):
	return completeness_score(labels_true, labels_pred)