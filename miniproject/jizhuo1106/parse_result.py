# This script will parse the output file and count detections

G_weap_count = 0
G_no_detections_count = 0
with open('result_gun.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        if 'no detections' in line:
            G_no_detections_count += 1
        elif ('gun' or 'knife') in line:
            G_weap_count += 1
print(f"Gun images with no detections: {G_no_detections_count}")
print(f"Gun images with weapons detected: {G_weap_count}")

K_weap_count = 0
K_no_detections_count = 0
with open('result_knife.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        if 'no detections' in line:
            K_no_detections_count += 1
        elif ('gun' or 'knife') in line:
            K_weap_count += 1
print(f"Knife images with no detections: {K_no_detections_count}")
print(f"knife images with weapons detected: {K_weap_count}")


N_weap_count = 0
N_no_detections_count = 0
with open('result_norm.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        if 'no detections' in line:
            N_no_detections_count += 1
        elif ('gun' or 'knife') in line:
            N_weap_count += 1
print(f"Normal images with no detections: {N_no_detections_count}")
print(f"Normal images with weapons detected: {N_weap_count}")

# 统计
true_positive = K_weap_count + G_weap_count
false_positive = N_weap_count
true_negative = N_no_detections_count
false_negative = G_no_detections_count + K_no_detections_count

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Calculate the confusion matrix
cm = np.array([[true_positive, false_negative],
               [false_positive, true_negative]])

# Plotting using seaborn
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Calculating metrics
accuracy = (true_positive + true_negative) / np.sum(cm)
recall = true_positive / (true_positive + false_negative)
precision = true_positive / (true_positive + false_positive)

# Print the statistics
print(f"Accuracy: {accuracy:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")