from matplotlib import pyplot as plt


def plot_roc_curve(fpr, tpr):
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange', label='ROC')  # The ROC curve
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # The random guess line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

