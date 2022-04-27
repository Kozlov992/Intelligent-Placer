import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#plt.rcParams['savefig.facecolor']='white'

def display_confusion_matrix(cf_matrix, name):
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    #accuracy  = np.trace(cf_matrix) / float(np.sum(cf_matrix))
    #stats_text = "\n\nAccuracy = {:0.3f}".format(accuracy)
    fig, ax = plt.subplots(1,1,dpi=250)
    hmap = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    lbs = [0, 1]
    hmap.set_xticklabels(lbs)
    hmap.set_yticklabels(lbs)
    #hmap.set_xlabel(stats_text, fontweight='bold')
    plt.yticks(rotation=0)
    #plt.xlabel(stats_text, fontweight='bold')
    plt.show()
    hmap.get_figure().savefig(name, facecolor='white')