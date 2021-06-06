from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


class AUC:
    def __init__(self):

        self.fig, self.axs = plt.subplots(2, 2)
        self.fig.suptitle('K Folds')
        self.j = 0
    def plot_roc_curve(self, fpr, tpr, auc):
        plt.title('Receiver Operating Characteristic')
        blue_patch = mpatches.Patch(color='blue', label='Mean AUC test = %0.2f' % auc)
        gray_patch = mpatches.Patch(color='gray', label='K-folds')
        red_patch = mpatches.Patch(color='red', label='Random Classifier', ls='--')
        plt.legend(handles=[blue_patch, gray_patch, red_patch], loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.plot(fpr, tpr, color='blue')
        plt.show()

    def plot_auc(self, fpr, tpr, auc, i):
        blue_patch = mpatches.Patch(color='blue', label='Mean AUC test = %0.2f' % auc)
        gray_patch = mpatches.Patch(color='gray', label='K-folds')
        red_patch = mpatches.Patch(color='red', label='Random Classifier', ls='--')
        self.axs[i, self.j].legend(handles=[blue_patch, gray_patch, red_patch], loc='lower right')
        self.axs[i, self.j].plot([0, 1], [0, 1], 'r--')
        self.axs[i, self.j].xlim([0, 1])
        self.axs[i, self.j].ylim([0, 1])
        self.axs[i, self.j].ylabel('True Positive Rate')
        self.axs[i, self.j].xlabel('False Positive Rate')
        self.axs[i, self.j].plot(fpr, tpr, color='blue')
        self.axs[i, self.j].show()

