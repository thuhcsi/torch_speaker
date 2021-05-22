from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y = [0,0,0,0,1,1,1,1]
score = [0.3,0.2,0.7,0.5,0.4,0.9,0.6,0.7]

fpr, tpr, threshold = roc_curve(y, score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('ROC Curve')
plt.show()
