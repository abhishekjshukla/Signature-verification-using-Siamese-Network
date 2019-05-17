import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score




def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))


md=load_model("sign.h5")


distances = [] 
identical = [] 

num = len(x_test)

for i in range(num):
    distances.append(distance(x_test[i][0], x_test[i][1]))
    identical.append(1)
    distances.append(distance(x_test[i][0], x_test[i][2]))
    identical.append(0)
        
distances = np.array(distances)
identical = np.array(identical)

thresholds = np.arange(0.001, .03, 0.002)

f1_scores = [f1_score(identical, distances < t) for t in thresholds]
acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

opt_idx = np.argmax(f1_scores)

opt_tau = thresholds[opt_idx]

opt_acc = accuracy_score(identical, distances < opt_tau)


plt.plot(thresholds, f1_scores, label='F1 score');
plt.plot(thresholds, acc_scores, label='Accuracy');
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')

plt.xlabel('Distance threshold')
plt.legend();

