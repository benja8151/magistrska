import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import os

def save_k_fold_results(num_folds, num_epochs, results, out_dir, out_name):
	actual_labels_combined = []
	predicted_labels_combined = []
	predicted_probabilities_combined = []

	actual_labels = results[0]['actual_labels_all']
	predicted_probabilities_averaged = []

	for fold in range(num_folds):

		actual_labels_combined = np.concatenate((actual_labels_combined, results[fold]["actual_labels_all"]))
		predicted_labels_combined = np.concatenate((predicted_labels_combined, results[fold]["predicted_labels_all"]))
		predicted_probabilities_combined = np.concatenate((predicted_probabilities_combined, results[fold]["predicted_probabilities_all"]))

		if len(predicted_probabilities_averaged) == 0:
			predicted_probabilities_averaged = np.array(results[fold]["predicted_probabilities_all"])
		else:
			predicted_probabilities_averaged += np.array(results[fold]["predicted_probabilities_all"])

	predicted_probabilities_averaged = predicted_probabilities_averaged / num_folds
	predicted_labels_averaged = np.rint(predicted_probabilities_averaged).astype(int)

	# Results with contatenated labels
	cm = confusion_matrix(actual_labels_combined, predicted_labels_combined)
	accuracy = (cm[0,0]+cm[1,1]) * 100/sum(sum(cm))
	sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
	specificity = cm[1,1]/(cm[1,0]+cm[1,1])
	precision = cm[0,0]/(cm[0,0]+cm[1,0])
	f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
	auc = roc_auc_score(actual_labels_combined, predicted_probabilities_combined)

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	f = open(f"{out_dir}/{out_name}", "w")
	f.write(f'Epochs: {num_epochs}\n')
	f.write(f'K-Folds: {num_folds}\n')
	f.write(f'Accuracy: {accuracy:.4f}\n')
	f.write(f'Specificity: {specificity:.4f}\n')
	f.write(f'Precision: {precision:.4f}\n')
	f.write(f'Sensitivity / Recall: {sensitivity:.4f}\n')
	f.write(f'F1 score: {f1:.4f}\n')
	f.write(f'AUC: {auc:.4f}\n')
	f.close()

	# Results with averaged probabilities
	cm = confusion_matrix(actual_labels, predicted_labels_averaged)
	accuracy = (cm[0,0]+cm[1,1]) * 100/sum(sum(cm))
	sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
	specificity = cm[1,1]/(cm[1,0]+cm[1,1])
	precision = cm[0,0]/(cm[0,0]+cm[1,0])
	f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
	auc = roc_auc_score(actual_labels, predicted_probabilities_averaged)

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	f = open(f"{out_dir}/{out_name}_averaged", "w")
	f.write(f'Epochs: {num_epochs}\n')
	f.write(f'K-Folds: {num_folds}\n')
	f.write(f'Accuracy: {accuracy:.4f}\n')
	f.write(f'Specificity: {specificity:.4f}\n')
	f.write(f'Precision: {precision:.4f}\n')
	f.write(f'Sensitivity / Recall: {sensitivity:.4f}\n')
	f.write(f'F1 score: {f1:.4f}\n')
	f.write(f'AUC: {auc:.4f}\n')
	f.close()


