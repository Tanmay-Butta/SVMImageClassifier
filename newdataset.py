import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from mpl_toolkits.mplot3d import Axes3D

# dir ="C:\\Users\\tanma\\AppData\\Local\\Programs\\Python\\Python37\\dataSet"
# data=[]
# c=['cat','random','raptile','dog']

# for cata in c:
#     path=os.path.join(dir,cata)
#     label=c.index(cata)

#     for img in os.listdir(path):
#         imgpath=os.path.join(path,img)
#         p_imag=cv2.imread(imgpath,0)
#         p_imag=cv2.resize(p_imag,(50,50))
#         image=np.array(p_imag).flatten()
#         data.append([image,label])

# print(len(data))
# pick_in=open("data1.pickle","wb")
# pickle.dump(data,pick_in)
# pick_in.close()
pick_in=open("data1.pickle","rb")
data=pickle.load(pick_in)
pick_in.close()
random.shuffle(data)
fs=[]
labels=[]

for f, label in data:
    fs.append(f)
    labels.append(label)

xtrain,xtest,ytrain,ytest=train_test_split(fs,labels,test_size=0.25)

# model = SVC(C=1, kernel="poly", gamma="auto")
# model.fit(xtrain, ytrain)
# pick=open("model3.sav","wb")
# pickle.dump(model,pick)
# pick.close()
pick=open("model3.sav","rb")
model=pickle.load(pick)
pick.close()
pre=model.predict(xtest)
ac=model.score(xtest,ytest)

c=['cat','random','raptile','dog']

print("Accuracy:",ac)
print("prediction:",c[pre[3]])
# Make predictions on the test set
y_pred = model.predict(xtest)

# Compute accuracy
accuracy = accuracy_score(ytest, y_pred)

# Generate classification report
class_report = classification_report(ytest, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(class_report)

mypet = xtest[3].reshape(50, 50)  # Reshape the image
mypet_rgb = cv2.cvtColor(mypet, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
plt.imshow(mypet_rgb)
plt.show()

cm = confusion_matrix(ytest, y_pred)
ax = sns.heatmap(cm, annot=True, fmt="d")
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

plt.show() 
precision, recall, f1_score, _ = precision_recall_fscore_support(ytest, y_pred, average='weighted') 
# Plot 3D visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(precision, recall, f1_score, c='blue', marker='o', s=100)

# Labels
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')
ax.set_zlabel('F1 Score')
ax.set_title('3D Visualization of Classification Metrics')

plt.show()





