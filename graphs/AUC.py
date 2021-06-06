from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


class AUC:
    def __init__(self):

        self.fig, self.axs = plt.subplots(2, 2)
        self.fig.suptitle('K Folds')
        self.index_map = {0: [0, 0], 1:[0,1],2:[1,0],3:[1,1]}

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

    def plot_auc(self, name, kfold_fpr_tpr, fpr, tpr, mean_auc, i):
        blue_patch = mpatches.Patch(color='blue', label='Mean AUC test = %0.2f' % mean_auc)
        gray_patch = mpatches.Patch(color='gray', label='K-folds')
        red_patch = mpatches.Patch(color='red', label='Random Classifier', ls='--')
        j, k = self.index_map[i]

        self.axs[j, k].legend(handles=[blue_patch, gray_patch, red_patch], loc='lower right', prop={'size': 6})
        self.axs[j, k].title.set_text(name)
        self.axs[j, k].plot([0, 1], [0, 1], 'r--')
        self.axs[j, k].set_xlim([0, 1])
        self.axs[j, k].set_ylim([0, 1])
        self.axs[j, k].set_ylabel('True Positive Rate')
        self.axs[j, k].set_xlabel('False Positive Rate')
        for fprs_tpr in kfold_fpr_tpr:
           self.axs[j, k].plot(fprs_tpr[0], fprs_tpr[1], color="gray")
        self.axs[j, k].plot(fpr, tpr, color='blue')


