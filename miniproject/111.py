import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Load the image to get the confusion matrix values
# img = mpimg.imread('/mnt/data/confusion_matrix.png')
#
# # Display the image for visual confirmation
# plt.imshow(img)
# plt.axis('off')  # Turn off axis
# plt.show()

# Values from the provided confusion matrix image
# True Positive (TP): 475 (norm predicted as norm)
# False Positive (FP): 48 (weap predicted as norm)
# False Negative (FN): 790 (norm predicted as weap)
# True Negative (TN): 731 (weap predicted as weap)

TP = 365
FP = 1000
FN = 0
TN = 779

# Calculate the total number of predictions
total = TP + FP + FN + TN

# Create the normalized confusion matrix
normalized_matrix = np.array([
    [TP, FP],
    [FN, TN]
]) / total

# Now we can plot the normalized matrix
fig, ax = plt.subplots()

# Create the heatmap for the normalized confusion matrix
cax = ax.matshow(normalized_matrix, cmap=plt.cm.Blues)

# Add color bar
plt.colorbar(cax)

# Add the normalized values as text in the center of the cells
for (i, j), val in np.ndenumerate(normalized_matrix):
    ax.text(j, i, f"{val:.3f}", ha='center', va='center', color='black' if normalized_matrix[i, j] < 0.5 else 'white')

# We don't have the labels from the image, but we'll assume they are the same as in the provided image.
ax.set_xticklabels(['', 'norm', 'weap'])
ax.set_yticklabels(['', 'norm', 'weap'])
ax.set_xlabel('True')
ax.set_ylabel('Predicted')
ax.set_title('Normalized Confusion Matrix')

# Remove the grid lines
ax.grid(False)

# Show the plot
plt.show()
