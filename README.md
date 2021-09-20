# Task2.ipynb
This project is about Prediction using unsupervised learning 


#Task 2 : Prediction using Unsupervised Machine Learning


""""In this K-means clustering task I've made an attempt to predict the optimum number of clusters and represent it visually from the given ‘Iris’ dataset.

Technical Stack : Scikit Learn, Numpy Array, Scipy, Pandas, Matplotlib""""


from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches
import sklearn.metrics as sm
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.cluster import DBSCAN 
from sklearn.decomposition import PCA



#Step 1 - Loading the dataset

iris = datasets.load_iris()
print(iris.data)



print(iris.target_names)


print(iris.target)


x = iris.data
y = iris.target


#Step 2 - Visualizing the input data and its Hierarchy

#Plotting the data
fig = plt.figure(1, figsize=(7,5))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(x[:, 3], x[:, 0], x[:, 2], edgecolor="k", s=50)
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
plt.title("Iris Clustering K Means=3", fontsize=14)
plt.show()

#Hierachy Clustering 
hier=linkage(x,"ward")
max_d=7.08
plt.figure(figsize=(15,8))
plt.title('Iris Hierarchical Clustering Dendrogram')
plt.xlabel('Species')
plt.ylabel('distance')
dendrogram(
    hier,
    truncate_mode='lastp',  
    p=50,                  
    leaf_rotation=90.,      
    leaf_font_size=8.,     
)
plt.axhline(y=max_d, c='k')
plt.show()


#Step 3 - Data Preprocessing

x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
y = pd.DataFrame(iris.target, columns=['Target'])

x.head()

y.head()

#Step4: Model training
iris_k_mean_model = KMeans(n_clusters=3)
iris_k_mean_model.fit(x)

print(iris_k_mean_model.labels_)

print(iris_k_mean_model.cluster_centers_)

#Step 5 - Visualizing the Model Cluster


plt.figure(figsize=(14,6))

colors = np.array(['red', 'green', 'blue'])

predictedY = np.choose(iris_k_mean_model.labels_, [1, 0, 2]).astype(np.int64)

plt.subplot(1, 2, 1)
plt.scatter(x['Petal Length'], x['Petal Width'], c=colors[y['Target']])
plt.title('Before classification')
red_patch = mpatches.Patch(color='red', label='Setosa')
green_patch = mpatches.Patch(color='green', label='Versicolor')
blue_patch=mpatches.Patch(color="blue", label='Virginica')
plt.legend(handles=[red_patch, green_patch, blue_patch])

    
plt.subplot(1, 2, 2)
plt.scatter(x['Petal Length'], x['Petal Width'], c=colors[predictedY])
plt.title("Model's classification")
plt.legend(handles=[red_patch, green_patch, blue_patch])




#Step 6 - Calculating the Accuracy and Confusion Matrix

sm.accuracy_score(predictedY, y['Target'])

sm.confusion_matrix(predictedY, y['Target'])


""""Conclusion
Finally i am able to successfully carry-out the prediction using Unsupervised Machine Learning task and was able to evaluate the model's clustering accuracy score.
Thank You"""""




















