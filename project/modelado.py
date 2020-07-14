#Autor: Alberto Calvo Madurga

import pandas as pd
import numpy as np
import csv
import re
import string
import time
import nltk
import sys
from nltk.corpus import stopwords
import statistics

import preprocessing_data as pre
#Warnings
import warnings;warnings.filterwarnings("ignore")

####### PARÁMETROS Y ARGUMENTOS ########
PATH="C:/Users/ASUS/Documents/tfg/tfg-acm-emorec/project/"
if len(sys.argv)==2:
    DATOS=sys.argv[1]
    PATH_IMAGES="C:/Users/ASUS/Documents/tfg/tfg-acm-emorec/project/datos_nuevos/"
    lista = pd.read_csv(PATH+DATOS)
else:
    DATOS="data/tweet15k_etiq.csv"
    PATH_IMAGES="C:/Users/ASUS/Documents/tfg/tfg-acm-emorec/project/data/"
    lista = pd.read_csv(PATH+DATOS)




# Preprocesamiento (de preprocessing_data.py)
t0=time.time()
preprocessed_tweets=[]
for i in range(0,len(lista)):
    preprocessed_tweets.append(pre.preprocessing_tweet_es(lista.Tweet[i]))
print("Tiempo preprocesado",round(time.time()-t0,2),"seg")

# Marcamos la información deseada
X = lista.iloc[:, 0].values #El texto
y = lista.iloc[:, 1].values #El signo


#------CONSTRUCCIÓN DEL SISTEMA------
#Extracción de características
from sklearn.feature_extraction.text import TfidfVectorizer

#Diseño del experimento
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

#Aloritmos de aprendizaje
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

#Selección de parámetros
from sklearn.model_selection import GridSearchCV

#Validación
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score,precision_score,recall_score

#Para seleccionar los parámetros
import itertools
import statistics

## Guardar los modelos
import joblib

############################################
## EXTRACCIÓN DE CARACTERÍSTICAS (TF-IDF) ##
############################################
MAXF=2500
tfidfconverter = TfidfVectorizer(max_features=MAXF,stop_words=stopwords.words('spanish'))
X = tfidfconverter.fit_transform(preprocessed_tweets).toarray()

###  Diseño 1: Hold Out 80/20 Estratificado con el conjunto test reservado del principio
print("-----Hold Out 80/20 estratificado con conjunto test reservado")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0,stratify=y)

#SVM
t0=time.time()
ajustefinal1=svm.SVC(kernel="linear",C=1.0)
ajustefinal1.fit(X_train,y_train)
predfinal1=ajustefinal1.predict(X_test)
print("Tiempo ajuste SVM:",round(time.time()-t0,2),"seg")

#Random Forest
t0=time.time()
ajustefinal2=RandomForestClassifier(n_estimators=1000,oob_score=True)
ajustefinal2.fit(X_train,y_train)
predfinal2=ajustefinal2.predict(X_test)

print("Tiempo ajuste Random Forest:",round(time.time()-t0,2),"seg")

#Regresión Logística
t0=time.time()
ajustefinal3=LogisticRegression(C=10.0)
ajustefinal3.fit(X_train,y_train)
predfinal3=ajustefinal3.predict(X_test)
print("Tiempo ajuste Regresión Logística:",round(time.time()-t0,2),"seg")


#Guardar los modelos
joblib.dump(ajustefinal1, PATH_IMAGES+'/modelos/ajuste_svm.pkl')
joblib.dump(ajustefinal2, PATH_IMAGES+'/modelos/ajuste_randomf.pkl')
joblib.dump(ajustefinal3, PATH_IMAGES+'/modelos/ajuste_reglog.pkl')


#Sistema de voto
predfinalvote=[]
for i in range(0,len(predfinal1)):
    predfinalvote.append(statistics.mode([predfinal1[i],predfinal2[i],predfinal3[i]]))

print("\n---SVM----\n")
print(classification_report(predfinal1, y_test))
print("\n---Random Forest----\n")
print(classification_report(predfinal2, y_test))
print("\n---Regresión Logística----\n")
print(classification_report(predfinal3, y_test))
print("\n---Voto por Mayoría----\n")
print(classification_report(predfinalvote, y_test))


###  Diseño 2: Validación Cruzada Estratificado k=10
FOLDS=10
print("----Validación cruzada estratificada de "+str(FOLDS)+" folds")
folds=StratifiedKFold(n_splits=FOLDS,shuffle=True,random_state=0)

t0=time.time()
tfidfconverter = TfidfVectorizer(max_features=MAXF,stop_words=stopwords.words('spanish'))
X = tfidfconverter.fit_transform(preprocessed_tweets).toarray()
print("Tiempo de construcción BoW:",round(time.time()-t0,2))

accuracy1=0
accuracy2=0
accuracy3=0
accuracyvote=0
precision1=0
precision2=0
precision3=0
precisionvote=0
recall1=0
recall2=0
recall3=0
recallvote=0
f1w1=0
f1w2=0
f1w3=0
f1wvote=0
for train_index,test_index in folds.split(X,y):
    t0=time.time()
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    ajuste1=svm.SVC(kernel="linear",C=1.0)
    ajuste1.fit(X_train,y_train)
    pred1=ajuste1.predict(X_test)

    ajuste2=RandomForestClassifier(n_estimators=1000,oob_score=True)
    ajuste2.fit(X_train,y_train)
    pred2=ajuste2.predict(X_test)

    ajuste3=LogisticRegression(C=10.0)
    ajuste3.fit(X_train,y_train)
    pred3=ajuste3.predict(X_test)

    predvote=[]
    for i in range(0,len(pred1)):
        predvote.append(statistics.mode([pred1[i],pred2[i],pred3[i]]))

    accuracy1+=round(accuracy_score(pred1, y_test),4)
    accuracy2+=round(accuracy_score(pred2, y_test),4)
    accuracy3+=round(accuracy_score(pred3, y_test),4)
    accuracyvote+=round(accuracy_score(predvote, y_test),4)

    precision1+=round(precision_score(pred1,y_test, average='weighted'),4)
    precision2+=round(precision_score(pred2,y_test, average='weighted'),4)
    precision3+=round(precision_score(pred3,y_test, average='weighted'),4)
    precisionvote+=round(precision_score(predvote,y_test, average='weighted'),4)

    recall1+=round(recall_score(pred1,y_test, average='weighted'),4)
    recall2+=round(recall_score(pred2,y_test, average='weighted'),4)
    recall3+=round(recall_score(pred3,y_test, average='weighted'),4)
    recallvote+=round(recall_score(predvote,y_test, average='weighted'),4)

    f1w1+=round(f1_score(pred1,y_test, average='weighted'),4)
    f1w2+=round(f1_score(pred2,y_test, average='weighted'),4)
    f1w3+=round(f1_score(pred3,y_test, average='weighted'),4)
    f1wvote+=round(f1_score(predvote,y_test, average='weighted'),4)
    print("Tiempo iteración:",round(time.time()-t0,2))

#Resultados diseño Validación cruzada
print("-----Tasa de aciertos-----")
print("SVM:",round(accuracy1*10,2))
print("Random Forest:",round(accuracy2*10,2))
print("Regresión Logística:",round(accuracy3*10,2))
print("Voto por mayoría:",round(accuracyvote*10,2),"\n")
print("-----Precission-----")
print("SVM:",round(precision1*10,2))
print("Random Forest:",round(precision2*10,2))
print("Regresión Logística:",round(precision3*10,2))
print("Voto por mayoría:",round(precisionvote*10,2),"\n")
print("-----Recall-----")
print("SVM:",round(recall1*10,2))
print("Random Forest:",round(recall2*10,2))
print("Regresión Logística:",round(recall3*10,2))
print("Voto por mayoría:",round(recallvote*10,2),"\n")
print("-----F1 Weighted-----")
print("SVM:",round(f1w1*10,2))
print("Random Forest:",round(f1w2*10,2))
print("Regresión Logística:",round(f1w3*10,2))
print("Voto por mayoría:",round(f1wvote*10,2),"\n")
