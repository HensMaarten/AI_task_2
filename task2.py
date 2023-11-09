import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

pd.options.mode.chained_assignment = None
    
# data (as pandas dataframes) 
studentsDF = pd.read_csv('./resources/students_dropout.csv',delimiter=";")
studentsDFPartial = studentsDF[['Course', 'Previous qualification', 'International', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)','Target']]
