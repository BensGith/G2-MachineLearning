from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ConfusionMatrix:
    def __init__(self, clf_name, true_labels, predicted_labels):
        self.clf_name = clf_name
        self.clf_cm = confusion_matrix(true_labels, predicted_labels)
        self.tn, self.fp, self.fn, self.tp = self.clf_cm.ravel()
        self.normalized_cm = self.clf_cm / float(len(true_labels))
        self.ax = plt.subplots(figsize=(4, 4))[1]

    def plot(self):
        c_map = sns.diverging_palette(225, 240, n=10, as_cmap=True)
        sns.heatmap(self.normalized_cm, annot=True, cmap=c_map, linewidths=3)
        # labels, title and ticks
        self.ax.set_xlabel('Predicted')
        self.ax.set_ylabel('True')
        self.ax.xaxis.set_ticklabels(['ZERO', 'ONE'])
        self.ax.yaxis.set_ticklabels(['ZERO', 'ONE'])
        self.ax.xaxis.set_ticks_position('top')
        self.ax.xaxis.set_label_position('top')
        plt.show()

