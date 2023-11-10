import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import streamlit as st

pd.options.mode.chained_assignment = None


def random_forest_classifier(estimators):
    # train the classifiers
    rf = RandomForestClassifier(criterion='entropy', n_estimators=estimators)
    rf1 = rf.fit(X_train,y_train.values.ravel())

    # create predictions
    predictionrf = rf1.predict(X_test)
    return predictionrf

def gaussian_naive_bayes_classifier():
    
    nb = GaussianNB()
    nb1 = nb.fit(X_train, y_train.values.ravel())

    # create predictions
    predictionnb = nb1.predict(X_test)
    return predictionnb

def knn_classifier(k):

    # I give the option to the user to set the k themselves, if they choose not to I use the optimal K
    if (k == None):
        # K-nearest neighbor algorithm
        k_values = [i for i in range (1,31)]
        scores = []
        #Get the Best Value of k
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            score = cross_val_score(knn, X_cat, y.values.ravel(), cv=5)
            scores.append(np.mean(score))

        best_index = np.argmax(scores)
        k = k_values[best_index]

    knn = KNeighborsClassifier(n_neighbors=k)
    knn1 = knn.fit(X_train, y_train.values.ravel())

    # create predictions
    predictionknn = knn1.predict(X_test)
    return predictionknn

def get_accuracy(prediction):
    #accuracy score
    return metrics.accuracy_score(y_test, prediction)

def get_confusion_matrix(prediction):
    # confusion matrix
    return confusion_matrix(y_test,prediction)
  
# data (as pandas dataframes) 
studentsDF = pd.read_csv('./resources/students_dropout.csv',delimiter=";")
studentsDFPartial = studentsDF[['Course', 'Previous qualification', 'International', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)','Target']]

#setup
course_mapping = {33 :  "Biofuel Production Technologies", 171 :  "Animation and Multimedia Design", 8014 :  "Social Service (evening attendance)", 9003 :  "Agronomy", 9070 :  "Communication Design", 9085 :  "Veterinary Nursing", 9119 : "Informatics Engineering", 9130 : "Equinculture", 9147 : "Management", 9238 : "Social Service", 9254 : "Tourism", 9500 : "Nursing", 9556 : "Oral Hygiene", 9670 : "Advertising and Marketing Management", 9773 : "Journalism and Communication", 9853 : "Basic Education", 9991 : "Management (evening attendance)"}
prev_qualification_mapping = {1 : "Secondary education", 2 : "Higher education - bachelor's degree", 3 : "Higher education - degree", 4 : "Higher education : master's", 5 : "Higher education : doctorate", 6 : "Frequency of higher education", 9 : "12th year of schooling : not completed", 10 : "11th year of schooling : not completed", 12 : "Other : 11th year of schooling", 14 : "10th year of schooling", 15 : "10th year of schooling : not completed", 19 : "Basic education 3rd cycle (9th/10th/11th year) or equiv.", 38 : "Basic education 2nd cycle (6th/7th/8th year) or equiv.", 39 : "Technological specialization course", 40 : "Higher education : degree (1st cycle)", 42 : "Professional higher technical course", 43 : "Higher education : master (2nd cycle)"	}
international_mapping = {1: "yes", 0: "no"}

studentsDFPartial['Course'] = studentsDFPartial['Course'].map(course_mapping)
studentsDFPartial['Previous qualification'] = studentsDFPartial['Previous qualification'].map(prev_qualification_mapping)
studentsDFPartial['International'] = studentsDFPartial['International'].map(international_mapping)



# categorial encoding
feature_cols = ['Course', 'Previous qualification', 'International', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']

X = studentsDFPartial[feature_cols]
y = studentsDFPartial[['Target']]

ce_ord = ce.OrdinalEncoder(cols = feature_cols)
X_cat = ce_ord.fit_transform(X)

# Data splicing
X_train, X_test, y_train, y_test = train_test_split(X_cat, y, test_size=0.2) # 80% training and 20% test

st.title("Task Machine Learning")
st.subheader("By Maarten Hens")

#eda
st.header('Sample Data')
st.write(studentsDFPartial.head(5))

st.header('Info')
st.write(studentsDFPartial.info())

st.header('Summary Statistics')
st.write(studentsDFPartial.describe())


st.header('Missing values:')
st.write(studentsDFPartial.isnull().sum())


# Visualize the distribution of a categorical variable
plt.figure(figsize=(10, 6))
sns.countplot(x='Target', data=studentsDFPartial)
plt.title('Results of students')
plt.ylabel('Count')
st.pyplot()


st.header('Select the number of trees you want to use in the random forest')
number_of_trees = st.slider('Select a value:', 50, 400, 100, step=5)

st.header('Select the number of neighbors you want to check in the KNN.')
st.text('If you do not touch the slider the optimal amount of neighbors will be chosen')
k_value = st.slider('Select a value:', 1, 31, None)

if st.button('Run predictions'):
    st.header("Random forest")
    pred_rf = random_forest_classifier(number_of_trees)
    st.text("Accuracy score")
    st.text(get_accuracy(pred_rf))
    st.text("Confustion Matrix")
    st.text(get_confusion_matrix(pred_rf))
    
    st.header("Gaussian Naive Bayes")
    pred_nb = gaussian_naive_bayes_classifier()
    st.text("Accuracy score")
    st.text(get_accuracy(pred_nb))
    st.text("Confustion Matrix")
    st.text(get_confusion_matrix(pred_nb))
    
    st.header("KNN")
    pred_knn = knn_classifier(k_value)
    st.text("Accuracy score")
    st.text(get_accuracy(pred_knn))
    st.text("Confustion Matrix")
    st.text(get_confusion_matrix(pred_knn))

