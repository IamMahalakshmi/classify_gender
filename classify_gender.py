from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

d_clf = tree.DecisionTreeClassifier()
svm_clf=svm.SVC()
knn_clf = KNeighborsClassifier(n_neighbors=3)


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
d_clf = d_clf.fit(X, Y)
svm_clf=svm_clf.fit(X,Y)
knn_clf=knn_clf.fit(X,Y)

test=[[190, 70, 43]]
print(test)

prediction_d_clf = d_clf.predict(test)
prediction_svm_clf=svm_clf.predict(test)
prediction_knn_clf=knn_clf.predict(test)


print("decision tree prediction" +str(prediction_d_clf))
print("SVM prediction" +str(prediction_svm_clf))
print(" KNN prediction"+str(prediction_knn_clf))

accuracy1=d_clf.score(test,prediction_d_clf)
accuracy2=svm_clf.score(test,prediction_svm_clf)
accuracy3=knn_clf.score(test,prediction_knn_clf)

print(accuracy1,accuracy2,accuracy3);
# CHALLENGE compare their reusults and print the best one!

