from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ConfusionMatrix:
    def __init__(self, clf_name, true_labels, predicted_labels):
        self.clf_name = clf_name
        self.clf_cm = confusion_matrix(true_labels, predicted_labels)
        self.tn, self.fp, self.fn, self.tp = self.clf_cm.ravel()
        self.normalized_cm = self.clf_cm / float(len(true_labels))
        self.ax = plt.subplots(figsize=(4, 4))[1]

    def plot(self):
        group_names = ['TN', 'FP', 'FN', 'TP']
        rates = self.normalized_cm.reshape(1, 4)[0]
        labels = ['{}\n{:.2f}'.format(group[0], group[1]) for group in zip(group_names, rates)]
        labels = np.asarray(labels).reshape(2, 2)
        sns.heatmap(self.normalized_cm, annot=labels, fmt='', cmap="Blues", linewidths=3)
        self.ax.set_xlabel('Predicted labels')
        self.ax.set_ylabel('True labels')
        self.ax.xaxis.set_ticklabels(['0', '1'])
        self.ax.yaxis.set_ticklabels(['0', '1'])
        self.ax.xaxis.set_ticks_position('top')
        self.ax.xaxis.set_label_position('bottom')
        plt.show()

