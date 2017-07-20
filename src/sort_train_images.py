import shutil
import pandas as pd
import numpy as np
train_labels = pd.read_csv('../data/train_labels.csv')
labels = np.array(train_labels.invasive.values[0:2295])

rp = np.random.permutation(len(labels))
r = [None] * len(labels)
for i in range(len(labels)):
        if np.nonzero(rp == i)[0][0] / float(len(labels)) < 0.2:
            r[i] = 1
        else:
            r[i] = 0

for i in range(len(labels)):
            if labels[i] == 0 and r[i] == 0:
                shutil.copyfile(("../data/train/" + str(i + 1) + ".jpg"),
                                "../data/tra/non_invasive/" + str(i + 1) + ".jpg")
            if labels[i] == 1 and r[i] == 0:
                shutil.copyfile(("../data/train/" + str(i + 1) + ".jpg"),
                                "../data/tra/invasive/" + str(i + 1) + ".jpg")
            if labels[i] == 0 and r[i] == 1:
                shutil.copyfile(("../data/train/" + str(i + 1) + ".jpg"),
                                "../data/val/non_invasive/" + str(i + 1) + ".jpg")
            if labels[i] == 1 and r[i] == 1:
                shutil.copyfile(("../data/train/" + str(i + 1) + ".jpg"),
                                "../data/val/invasive/" + str(i + 1) + ".jpg")
