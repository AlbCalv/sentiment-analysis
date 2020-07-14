#Autor: Alberto Calvo Madurga

import json
import csv
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import string
import nltk
from statistics import mode, StatisticsError
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings;warnings.filterwarnings("ignore")

import preprocessing_data as pre

####### PARÁMETROS Y ARGUMENTOS ########
PATH="C:/Users/ASUS/Documents/tfg/tfg-acm-emorec/project/"
if len(sys.argv)==2:
    DATOS=sys.argv[1]
    DATOS=DATOS[:-4]
    PATH_IMAGES="C:/Users/ASUS/Documents/tfg/tfg-acm-emorec/project/datos_nuevos/"
else:
    DATOS="data/tweet15k"
    PATH_IMAGES="C:/Users/ASUS/Documents/tfg/tfg-acm-emorec/project/data/"

################################################################################
print("-------ETIQUETADO DE LOS DATOS------\n")
## lectura
t0=time.time()
lista = pd.read_csv(PATH+DATOS+".csv")
lista["posemo"] = 0.
lista["negemo"] = 0.
lista["et_spl"] = 0.
lista["et_senticon"] = 0.
lista["et_isol"] = 0.
lista["et_crisol"] = 0.
lista["etiqueta"] = 5.
print("Tiempo cargado:",round(time.time()-t0,2),"seg")

## PREPROCESAMIENTO CON LEMATIZACIÓN
tinicial=time.time()
# Preprocesamiento (de preprocessing_data.py)
t0=time.time()
preprocessed_tweets=[]
for i in range(0,len(lista)):
    preprocessed_tweets.append(pre.preprocessing_tweet_es(lista.Tweet[i]))
print("Tiempo preprocesado:",round(time.time()-t0,2),"seg\n")

## DETECCIÓN DE EMOJIS
print("------LÉXICO DE EMOJIS-----\n")

#Listado de emojis con su polaridad asociada  http://kt.ijs.si/data/Emoji_sentiment_ranking/
listaemojis=pd.read_csv(PATH+"lexicos/emoji/Emoji Sentiment Ranking 1.0.csv")
listaemojis=listaemojis[listaemojis.Occurrences>5]
listaemojis.Negative=listaemojis.Negative/listaemojis.Occurrences
listaemojis.Positive=listaemojis.Positive/listaemojis.Occurrences
listaemojis.Neutral=listaemojis.Neutral/listaemojis.Occurrences

import emoji
emojis=[]
## Proceso común para crisol y mlsenticon
t0=time.time()
for i in range(0,len(lista)):
    emojis=pre.text_has_emoji(preprocessed_tweets[i])
    for e in emojis:
        if listaemojis['Emoji'].eq(e).any():
            index=listaemojis[listaemojis['Emoji']==e].index.item()
            lista.posemo[i]=1.0*listaemojis.Positive[index]
            lista.negemo[i]=-1.0*listaemojis.Negative[index]
print("Tiempo emojis 1:",round(time.time()-t0,2),"seg")

###############################################
##------Sentiment Polarity Lexicon (es)------##
###############################################
print("\n------LÉXICO SPL-----\n")
t0=time.time()
negative_words = pd.read_csv(PATH+'lexicos/spl/negative_words_es.txt', sep="\n", names=['WORD'])
positive_words = pd.read_csv(PATH+'lexicos/spl/positive_words_es.txt', sep="\n", names=['WORD'])
negative_words.WORD=negative_words.WORD.str.rstrip()
negative_words.WORD=negative_words.WORD.str.lstrip()
positive_words.WORD=positive_words.WORD.str.rstrip()
positive_words.WORD=positive_words.WORD.str.lstrip()
print("Tiempo de cargado:",round(time.time()-t0,2),"seg")

t0=time.time()
positive_emojis_spl=np.zeros(len(lista),dtype=int)
negative_emojis_spl=np.zeros(len(lista),dtype=int)
emojis=[]
for i in range(0,len(lista)):
    emojis=pre.text_has_emoji(preprocessed_tweets[i])
    for e in emojis:
        if listaemojis['Emoji'].eq(e).any():
            index=listaemojis[listaemojis['Emoji']==e].index.item()
            if listaemojis.Positive[index]>listaemojis.Negative[index]:
                positive_emojis_spl[i]+=1
            elif listaemojis.Positive[index]<listaemojis.Negative[index]:
                negative_emojis_spl[i]+=1
print("Tiempo emojis 2:",round(time.time()-t0,2),"seg")

## Cálculo terminos positivos y negativos
positive_terms_spl=np.zeros(len(lista),dtype=int)
negative_terms_spl=np.zeros(len(lista),dtype=int)
#Para el conteo de palabras repetidas
index_positive_spl=np.zeros(len(positive_words),dtype=int)
index_negative_spl=np.zeros(len(negative_words),dtype=int)
t0=time.time()
for num in range(0,len(lista)):
    #Eliminación de StopWords
    for palabra in [word for word in word_tokenize(preprocessed_tweets[num]) if word not in stopwords.words('spanish')]:
        if positive_words['WORD'].eq(palabra).any():
            index=positive_words[positive_words['WORD']==palabra].index.item()
            index_positive_spl[index]+=1
            positive_terms_spl[num]+=1
        elif negative_words['WORD'].eq(palabra).any():
            index=negative_words[negative_words['WORD']==palabra].index.item()
            index_negative_spl[index]+=1
            negative_terms_spl[num]+=1
print("Tiempo cálculo términos positivos y negativos:",round(time.time()-t0,2),"seg\n")

## Análisis de términos
total_pos_spl=len(positive_words)
total_neg_spl=len(negative_words)
unicos_pos_spl=len(index_positive_spl[index_positive_spl>0])
unicos_neg_spl=len(index_negative_spl[index_negative_spl>0])
term_pos_spl=sum(index_positive_spl)
term_neg_spl=sum(index_negative_spl)
term_spl=term_pos_spl+term_neg_spl

print("Términos positivos únicos encontrados:",unicos_pos_spl," de ",total_pos_spl,"(",round(unicos_pos_spl/total_pos_spl*100,2),"%)")
print("Términos negativos únicos encontrados:",unicos_neg_spl," de ",total_neg_spl,"(",round(unicos_neg_spl/total_neg_spl*100,2),"%)")
print("Términos positivos totales encontrados:",term_pos_spl)
print("Términos negativos totales encontrados:",term_neg_spl)
print("Términos totales encontrados:",term_spl)

## Gráficos de barras
bars=list(positive_words.WORD)
freq=list(index_positive_spl)
df = pd.DataFrame({"Palabras":bars,
                  "Frecuencia":freq})
df.sort_values('Frecuencia',inplace=True,ascending=False)
df=df[0:10]
#Gráfico
plt.figure(figsize=(10,6))
plt.bar('Palabras', 'Frecuencia',data=df,color="green")
plt.title("Frecuencia palabras positivas - SPL", size=18)
plt.savefig(PATH_IMAGES+'/analitics/freqposspl.png')

bars=list(negative_words.WORD)
freq=list(index_negative_spl)
df = pd.DataFrame({"Palabras":bars,
                  "Frecuencia":freq})
df.sort_values('Frecuencia',inplace=True,ascending=False)
df=df[0:10]
#Gráfico
plt.figure(figsize=(10,6))
plt.bar('Palabras', 'Frecuencia',data=df,color="red")
plt.title("Frecuencia palabras negativas -SPL", size=18)
plt.savefig(PATH_IMAGES+'/analitics/freqnegspl.png')

## Etiquetado de los datos
# Si todos los valores son 0, es neutro
t0=time.time()
neutros_spl=0
for i in range(0,len(lista)):
    if positive_terms_spl[i] == negative_terms_spl[i] and positive_emojis_spl[i] == negative_emojis_spl[i]:
        lista.et_spl[i]=0
        if positive_terms_spl[i]==0 and positive_emojis_spl[i]==0:
            neutros_spl+=1
    else:
        #Para etiquetar la polaridad positiva o negativa max(|positiveSent+positiveEmoji|,|negativeSent+negativeEmoji|)
        if np.abs(positive_terms_spl[i]+positive_emojis_spl[i])>np.abs(negative_terms_spl[i]+negative_emojis_spl[i]):
            lista.et_spl[i]=1
        else:
            lista.et_spl[i]=-1
print("Número de tweets con 0 polaridad:",neutros_spl,"\n")
print("Tiempo etiquetado:",round(time.time()-t0,2),"seg\n")


print("--Etiquetado según léxico SPL--")
print(lista.et_spl.value_counts())

##############################
##---------ML-SentiCon------##
##############################
print("\n------LÉXICO SENTICON-----\n")
t0=time.time()
senticon = pd.read_excel(PATH+'lexicos/mlsenticon/senticon_limpio.xlsx',usecols=[2,3,4])
senticon=senticon.rename(columns = {"Unnamed: 2":"Nivel","Unnamed: 3":"Palabra","Unnamed: 4":"Polarity"})
senticon.iloc[3417,1]="false" #Daban problemas porque estaban como True y False
senticon.iloc[6264,1]="true"
senticon=senticon[1:11543] #La primera línea no tiene nada
senticon=senticon.reset_index(drop=True)
senticon=senticon.replace(to_replace = '_', value = ' ',regex=True) #Convertir barrabaja en espacio para poder eliminar las multi
#Solo 1 palabra
for i in range(0,len(senticon)):
    if len(senticon.Palabra[i].split())>1:
        senticon = senticon.drop(i, axis=0)
senticon=senticon.reset_index(drop=True)
senticon.drop_duplicates(subset ="Palabra", keep = "first",inplace=True)
senticon=senticon.reset_index(drop=True)
senticon.Palabra=senticon.Palabra.str.rstrip()
senticon.Palabra=senticon.Palabra.str.lstrip()
print("Tiempo cargado:",round(time.time()-t0,2),"seg")

## Cálculo máximo término positivo y negativo por tweet
positive_terms_senticon=np.zeros(len(lista))
negative_terms_senticon=np.zeros(len(lista))
#Para el conteo de palabras repetidas
index_positive_senticon=np.zeros(len(senticon),dtype=int)
index_negative_senticon=np.zeros(len(senticon),dtype=int)
t0=time.time()
for num in range(0,len(lista)):
    #Eliminación de StopWords
    for palabra in [word for word in word_tokenize(preprocessed_tweets[num]) if word not in stopwords.words('spanish')]:
        if senticon['Palabra'].eq(palabra).any():
            index=senticon[senticon['Palabra']==palabra].index.item()
            if senticon.Polarity[index]>0 and senticon.Polarity[index]>positive_terms_senticon[num]:
                positive_terms_senticon[num]=senticon.Polarity[index]
                index_positive_senticon[index]+=1
            elif senticon.Polarity[index]<0 and senticon.Polarity[index]<negative_terms_senticon[num]:
                negative_terms_senticon[num]=senticon.Polarity[index]
                index_negative_senticon[index]+=1
print("Tiempo de cálculo de máximo término positivo y negativo:",round(time.time()-t0,2),"seg")

## Análisis de términos
total_pos_senticon=len(senticon[senticon.Polarity>0])
total_neg_senticon=len(senticon[senticon.Polarity<0])
unicos_pos_senticon=len(index_positive_senticon[index_positive_senticon>0])
unicos_neg_senticon=len(index_negative_senticon[index_negative_senticon>0])
term_pos_senticon=sum(index_positive_senticon)
term_neg_senticon=sum(index_negative_senticon)
term_senticon=term_pos_senticon+term_neg_senticon

print("Términos positivos únicos encontrados:",unicos_pos_senticon," de ",total_pos_senticon,"(",round(unicos_pos_senticon/total_pos_senticon,4)*100,"%)")
print("Términos negativos únicos encontrados:",unicos_neg_senticon," de ",total_neg_senticon,"(",round(unicos_neg_senticon/total_neg_senticon,4)*100,"%)")
print("Términos positivos totales encontrados:",term_pos_senticon)
print("Términos negativos totales encontrados:",term_neg_senticon)
print("Términos totales encontrados:",term_senticon)

## Gráficos de barras 1
bars=list(senticon.Palabra)
freq=list(index_positive_senticon)
df = pd.DataFrame({"Palabras":bars,
                  "Frecuencia":freq,
                  "Grado":senticon.Polarity})
df.sort_values('Frecuencia',inplace=True,ascending=False)
df=df[0:10]

likeability_scores = np.array(df.Grado)
data_normalizer = mpl.colors.Normalize()
color_map = mpl.colors.LinearSegmentedColormap(
    "my_map",
    {
        "red": [(0, 0.5, 0.5),
                (1.0, 0.0, 0.0)],
        "green": [(0, 1.0, 1.0),
                  (1.0, 0.5, 0.5)],
        "blue": [(0, 0.50, 0.5),
                 (1.0, 0, 0)]
    }
)
#Gráfico
plt.figure(figsize=(10,6))
plt.bar('Palabras', 'Frecuencia',data=df,color=color_map(data_normalizer(likeability_scores)))
plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=mpl.cm.Greens),
             orientation='vertical', label='Polaridad  Positiva')
plt.title("Frecuencia palabras positivas - SentiCon", size=18)
plt.savefig(PATH_IMAGES+'/analitics/freqpossenticon.png')

## Gráficos de barras 2
bars=list(senticon.Palabra)
freq=list(index_negative_senticon)
df = pd.DataFrame({"Palabras":bars,
                  "Frecuencia":freq,
                  "Grado":senticon.Polarity})
df.sort_values('Frecuencia',inplace=True,ascending=False)
df=df[0:10]

likeability_scores = np.array(df.Grado)
data_normalizer = mpl.colors.Normalize()
color_map = mpl.colors.LinearSegmentedColormap(
    "my_map",
    {
        "red": [(0, 1.0, 1.0),
                (1.0, .5, .5)],
        "green": [(0, 0.5, 0.5),
                  (1.0, 0, 0)],
        "blue": [(0, 0.50, 0.5),
                 (1.0, 0, 0)]
    }
)
#Gráfico
plt.figure(figsize=(15,6))
plt.bar('Palabras', 'Frecuencia',data=df,color=color_map(data_normalizer(likeability_scores)))
plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=mpl.cm.Reds),
             orientation='vertical', label='Polaridad  Negativa')
plt.title("Frecuencia palabras negativas - SentiCon", size=18)
plt.savefig(PATH_IMAGES+'/analitics/freqnegsenticon.png')

## Etiquetado de cada tweet
# Si todos los valores son 0, es neutro
t0=time.time()
neutros_senticon=0
for i in range(0,len(lista)):
    if np.abs(positive_terms_senticon[i]-negative_terms_senticon[i])<0.02 and np.abs(lista.posemo[i]-lista.negemo[i])<0.02:
        lista.et_senticon[i]=0
        if positive_terms_senticon[i]==0 and negative_terms_senticon[i]==0 and lista.posemo[i]==0 and lista.negemo[i]==0:
            neutros_senticon+=1
    else:
        #Para etiquetar la polaridad positiva o negativa max(|positiveSent+positiveEmoji|,|negativeSent+negativeEmoji|)
        if np.abs(positive_terms_senticon[i]+lista.posemo[i])>np.abs(negative_terms_senticon[i]+lista.negemo[i]):
            lista.et_senticon[i]=1
        else:
            lista.et_senticon[i]=-1
print("Número de tweets con 0 polaridad:",neutros_senticon,"\n")
print("Tiempo de etiquetado Senticon:",round(time.time()-t0,2),"seg\n")
print(lista.et_senticon.value_counts())

##############################
##----------iSol------------##
##############################
print("\n------LÉXICO ISOL-----\n")
positive_emojis_isol=np.zeros(len(lista),dtype=int)
negative_emojis_isol=np.zeros(len(lista),dtype=int)
emojis=[]
t0=time.time()
for i in range(0,len(lista)):
    emojis=pre.text_has_emoji(preprocessed_tweets[i])
    for e in emojis:
        if listaemojis['Emoji'].eq(e).any():
            index=listaemojis[listaemojis['Emoji']==e].index.item()
            if listaemojis.Positive[index]>listaemojis.Negative[index]:
                positive_emojis_isol[i]+=1
            elif listaemojis.Positive[index]<listaemojis.Negative[index]:
                negative_emojis_isol[i]+=1
print("Tiempo emojis isol:",round(time.time()-t0,2),"seg")

## Eliminación palabras repetidas y expresiones del léxico
t0=time.time()
isol=pd.read_excel(PATH+'lexicos/crisol/isol.xlsx',header=None,names=["Palabra", "Polaridad"])
#Solo 1 palabra
for i in range(0,len(isol)):
    if len(isol.Palabra[i].split())>1:
        isol = isol.drop(i, axis=0)
isol=isol.reset_index(drop=True) #Todas son de una palabra
#Hay duplicados con signo positivos (partidarios y simplificada) y duplicados con mismo signo (daños y engaños)
#Los de signo contrario los eliminamos, los de mismo signo eliminamos uno de ellos
isol=isol.drop(1790)
isol=isol.drop(2282)
isol=isol.drop(6814)
isol=isol.drop(7584)
isol=isol.reset_index(drop=True)
isol.drop_duplicates(subset ="Palabra", keep = "first",inplace=True)
isol=isol.reset_index(drop=True)
isol.Palabra=isol.Palabra.str.rstrip()
isol.Palabra=isol.Palabra.str.lstrip()
print("Tiempo de cargado:",round(time.time()-t0,2),"seg")

## Cálculo terminos positivos y negativos
positive_terms_isol=np.zeros(len(lista),dtype=int)
negative_terms_isol=np.zeros(len(lista),dtype=int)
#Para el conteo de palabras repetidas
index_positive_isol=np.zeros(len(isol),dtype=int)
index_negative_isol=np.zeros(len(isol),dtype=int)
t0=time.time()
for num in range(0,len(lista)):
    #Eliminación de StopWords
    for palabra in [word for word in word_tokenize(preprocessed_tweets[num]) if word not in stopwords.words('spanish')]:
        if isol['Palabra'].eq(palabra).any():
            index=isol[isol['Palabra']==palabra].index.item()
            if isol.Polaridad[index]=="positive":
                positive_terms_isol[num]+=1
                index_positive_isol[index]+=1
            elif isol.Polaridad[index]=="negative":
                negative_terms_isol[num]+=1
                index_negative_isol[index]+=1
print("Tiempo de cálculo de términos positivo y negativo:",round(time.time()-t0,2),"seg\n")

## Análisis de Términos
total_pos_isol=len(isol[isol.Polaridad=="positive"])
total_neg_isol=len(isol[isol.Polaridad=="negative"])
unicos_pos_isol=len(index_positive_isol[index_positive_isol>0])
unicos_neg_isol=len(index_negative_isol[index_negative_isol>0])
term_pos_isol=sum(index_positive_isol)
term_neg_isol=sum(index_negative_isol)
term_isol=term_pos_isol+term_neg_isol

print("Términos positivos únicos encontrados:",unicos_pos_isol," de ",total_pos_isol,"(",round(unicos_pos_isol/total_pos_isol,4)*100,"%)")
print("Términos negativos únicos encontrados:",unicos_neg_isol," de ",total_neg_isol,"(",round(unicos_neg_isol/total_neg_isol,4)*100,"%)")
print("Términos positivos totales encontrados:",term_pos_isol)
print("Términos negativos totales encontrados:",term_neg_isol)
print("Términos totales encontrados:",term_isol)

## Gráficos de barras 1
bars=list(isol.Palabra)
freq=list(index_positive_isol)
df = pd.DataFrame({"Palabras":bars,
                  "Frecuencia":freq})
df.sort_values('Frecuencia',inplace=True,ascending=False)
df=df[0:10]
#Gráfico
plt.figure(figsize=(10,6))
plt.bar('Palabras', 'Frecuencia',data=df,color="green")
plt.title("Frecuencia palabras positivas - iSOL", size=18)
plt.savefig(PATH_IMAGES+'/analitics/freqposisol.png')

## Gráficos de barras 2
bars=list(isol.Palabra)
freq=list(index_negative_isol)
df = pd.DataFrame({"Palabras":bars,
                  "Frecuencia":freq})
df.sort_values('Frecuencia',inplace=True,ascending=False)
df=df[0:10]
#Gráfico
plt.figure(figsize=(10,6))
plt.bar('Palabras', 'Frecuencia',data=df,color="red")
plt.title("Frecuencia palabras negativas - iSOL", size=18)
plt.savefig(PATH_IMAGES+'/analitics/freqnegisol.png')

## Etiquetado de cada tweet
# Si todos los valores son 0, es neutro
t0=time.time()
neutros_isol=0
for i in range(0,len(lista)):
    if positive_terms_isol[i] == negative_terms_isol[i] and positive_emojis_isol[i] == negative_emojis_isol[i]:
        lista.et_isol[i]=0
        if positive_terms_isol[i]==0 and negative_terms_isol[i]==0 and positive_emojis_isol[i]==0 and negative_emojis_isol[i]==0:
            neutros_isol+=1
    else:
        #Para etiquetar la polaridad positiva o negativa max(|positiveSent+positiveEmoji|,|negativeSent+negativeEmoji|)
        if np.abs(positive_terms_isol[i]+positive_emojis_isol[i])>np.abs(negative_terms_isol[i]+negative_emojis_isol[i]):
            lista.et_isol[i]=1
        else:
            lista.et_isol[i]=-1
print("Número de tweets con 0 polaridad:",neutros_isol,"\n")
print("Tiempo de etiquetado ISOL:",round(time.time()-t0,2),"seg\n")
print(lista.et_isol.value_counts())

##############################
##---------CRiSol-----------##
##############################
print("\n------LÉXICO CRISOL-----\n")
t0=time.time()
crisol=pd.read_excel(PATH+'lexicos/crisol/crisol.xlsx',header=None,names=["ID","Palabra","Proc","Pos","Polarity","Positive","Neutral","Negative"])
crisol.drop_duplicates(subset="ID",keep="last",inplace=True) #Eliminación de las repetidas (isol)
crisol=crisol[crisol.Pos!="x"] #Las que tienen la x son las que tienen categoría morfológica y puntuaciones de polaridad
crisol=crisol.reset_index(drop=True)
crisol=crisol.drop(1066)
crisol=crisol.drop(3727)
crisol=crisol.reset_index(drop=True)
crisol.drop_duplicates(subset="Palabra",keep="last",inplace=True) #Eliminación de las repetidas (isol)
crisol=crisol.reset_index(drop=True)
crisol.Palabra=crisol.Palabra.str.rstrip()
crisol.Palabra=crisol.Palabra.str.lstrip()
print("Tiempo cargado:",round(time.time()-t0,2),"seg")

## Cálculo del máximo término positivo y negativo por tweet
positive_terms_crisol=np.zeros(len(lista))
negative_terms_crisol=np.zeros(len(lista))
#Para el conteo de palabras repetidas
index_positive_crisol=np.zeros(len(crisol),dtype=int)
index_negative_crisol=np.zeros(len(crisol),dtype=int)
t0=time.time()
for num in range(0,len(lista)):
    indexmax=-1
    indexmin=-1
    #Eliminación de StopWords
    for palabra in [word for word in word_tokenize(preprocessed_tweets[num]) if word not in stopwords.words('spanish')]:
        if crisol['Palabra'].eq(palabra).any():
            index=crisol[crisol['Palabra']==palabra].index.item()
            if crisol.Positive[index]>positive_terms_crisol[num]:
                indexmax=index
                positive_terms_crisol[num]=crisol.Positive[index]
            if crisol.Negative[index]*-1<negative_terms_crisol[num]:
                indexmin=index
                negative_terms_crisol[num]=-1*crisol.Negative[index]
    if indexmax>-1:
        index_positive_crisol[indexmax]+=1
    if indexmin>-1:
        index_negative_crisol[indexmin]+=1
print("Tiempo cálculo máximo término positivo y negativo:",round(time.time()-t0,2),"seg\n")

## Análisis de Términos
total_pos_crisol=len(crisol[crisol.Polarity=="positive"])
total_neg_crisol=len(crisol[crisol.Polarity=="negative"])
unicos_pos_crisol=len(index_positive_crisol[index_positive_crisol>0])
unicos_neg_crisol=len(index_negative_crisol[index_negative_crisol>0])
term_pos_crisol=sum(index_positive_crisol)
term_neg_crisol=sum(index_negative_crisol)
term_crisol=term_pos_crisol+term_neg_crisol

print("Términos positivos únicos encontrados:",unicos_pos_crisol," de ",total_pos_crisol,"(",round(unicos_pos_crisol/total_pos_crisol,4)*100,"%)")
print("Términos negativos únicos encontrados:",unicos_neg_crisol," de ",total_neg_crisol,"(",round(unicos_neg_crisol/total_neg_crisol,4)*100,"%)")
print("Términos positivos totales encontrados:",term_pos_crisol)
print("Términos negativos totales encontrados:",term_neg_crisol)
print("Términos totales encontrados:",term_crisol)

## Gráfico de barras 1
bars=list(crisol.Palabra)
freq=list(index_positive_crisol)
df = pd.DataFrame({"Palabras":bars,
                  "Frecuencia":freq,
                  "Grado":crisol.Positive})
df.sort_values('Frecuencia',inplace=True,ascending=False)
df=df[0:10]

likeability_scores = np.array(df.Grado)
data_normalizer = mpl.colors.Normalize()
color_map = mpl.colors.LinearSegmentedColormap(
    "my_map",
    {
        "red": [(0, 0.5, 0.5),
                (1.0, 0.0, 0.0)],
        "green": [(0, 1.0, 1.0),
                  (1.0, 0.5, 0.5)],
        "blue": [(0, 0.50, 0.5),
                 (1.0, 0, 0)]
    }
)
#Gráfico
plt.figure(figsize=(10,5))
plt.bar('Palabras', 'Frecuencia',data=df,color=color_map(data_normalizer(likeability_scores)))
plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=mpl.cm.Greens),
             orientation='vertical', label='Polaridad  Positiva')
plt.title("Frecuencia palabras positivas - CRiSOL", size=18)
plt.savefig(PATH_IMAGES+'/analitics/freqposcrisol.png')

## Gráfico de barras 2
bars=list(crisol.Palabra)
freq=list(index_negative_crisol)
df = pd.DataFrame({"Palabras":bars,
                  "Frecuencia":freq,
                  "Grado":crisol.Negative})
df.sort_values('Frecuencia',inplace=True,ascending=False)
df=df[0:10]

likeability_scores = np.array(df.Grado)
data_normalizer = mpl.colors.Normalize()
color_map = mpl.colors.LinearSegmentedColormap(
    "my_map",
    {
        "red": [(0, 1.0, 1.0),
                (1.0, 0.5, 0.5)],
        "green": [(0, 0.5, 0.5),
                  (1.0, 0.0, 0.0)],
        "blue": [(0, 0.50, 0.5),
                 (1.0, 0, 0)]
    }
)
#Gráfico
plt.figure(figsize=(10,5))
plt.bar('Palabras', 'Frecuencia',data=df,color=color_map(data_normalizer(likeability_scores)))
plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=mpl.cm.Reds),
             orientation='vertical', label='Polaridad  Negativa')
plt.title("Frecuencia palabras negativas - CRiSOL", size=18)
plt.savefig(PATH_IMAGES+'/analitics/freqnegcrisol.png')

## Etiquetado de los tweets
# Si todos los valores son 0, es neutro
t0=time.time()
neutros_crisol=0
for i in range(0,len(lista)):
    if negative_terms_crisol[i] == positive_terms_crisol[i] and lista.posemo[i] == lista.negemo[i]:
        lista.et_crisol[i]=0
        if positive_terms_crisol[i]==0 and negative_terms_crisol[i]==0 and lista.posemo[i]==0 and lista.negemo[i]==0:
            neutros_crisol+=1
    else:
        #Para etiquetar la polaridad positiva o negativa max(|positiveSent+positiveEmoji|,|negativeSent+negativeEmoji|)
        if np.abs(positive_terms_crisol[i]+lista.posemo[i])>np.abs(negative_terms_crisol[i]+lista.negemo[i]):
            lista.et_crisol[i]=1
        else:
            lista.et_crisol[i]=-1
print("Número de tweets con 0 polaridad:",neutros_crisol,"\n")
print("Tiempo de etiquetado CRISOL:",round(time.time()-t0,2),"seg\n")
print(lista.et_crisol.value_counts())

######################################
## Estadísticas de Términos Finales ##
######################################
print("\n------ESTADÍSTICAS DE TÉRMINOS -----\n")
print("Palabras Totales")
print("SLP:",term_spl)
print("ML-SentiCon:",term_senticon)
print("iSOL:",term_isol)
print("CRiSOL:",term_crisol)
print("Total:",term_crisol+term_isol+term_spl+term_senticon)
print("")

print("Palabras Positivas")
print("SLP:",term_pos_spl)
print("ML-SentiCon:",term_pos_senticon)
print("iSOL:",term_pos_isol)
print("CRiSOL:",term_pos_crisol)
print("Total:",term_pos_crisol+term_pos_isol+term_pos_spl+term_pos_senticon)
print("")

print("Palabras Negativas")
print("SLP:",term_neg_spl)
print("ML-SentiCon:",term_neg_senticon)
print("iSOL:",term_neg_isol)
print("CRiSOL:",term_neg_crisol)
print("Total:",term_neg_crisol+term_neg_isol+term_neg_spl+term_neg_senticon)
print("")

##############################
##      Etiquetado final    ##
##############################
print("\n------ETIQUETADO FINAL-----\n")
t0=time.time()
for i in range(0,len(lista)):
    try:
        lista.etiqueta[i]=mode(lista.iloc[i][3:7])
    except StatisticsError:
        #Se pueden dar tres casos de igualdad de modas:
        #  2 etiquetan 1 y 2 etiquetan 0 --> Et.Final=Positivo
        #  2 etiquetan -1 y 2 etiquetan 0 --> Et.Final=Negativo
        #  2 etiquetan 1 y 2 etiquetean 1 --> Et.Final=Neutro
        media=np.mean(lista.iloc[i,3:7])
        if media > 0:
            lista.etiqueta[i]=1.0
        elif media < 0:
            lista.etiqueta[i]=-1.0
        elif media==0:
            lista.etiqueta[i]=0.0
print("Tiempo Etiquetado Final:",round(time.time()-t0,2),"seg\n")
print(lista.etiqueta.value_counts())
tfinal=time.time()-tinicial #Tiempo de todo el sistema
print("\n**Tiempo total del sistema:",round(tfinal,2),"seg\n")
lista.to_csv(PATH+DATOS+'_etiq_neutro.csv',columns=['Tweet','etiqueta'],index=False)  #Descomentar para guardar con neutros


## Creación de archivos de tweets nulos y no nulos
nulos=lista[(lista['et_spl']==0.0) & (lista['et_senticon']==0.0)& (lista['et_isol']==0.0)& (lista['et_crisol']==0.0)]
nulos=nulos.reset_index(drop=True)
#nulos.to_csv(PATH+'/es/prueba_tweet15k_nulos.csv',columns=['Tweet','etiqueta'],index=False)
nulos.to_csv(PATH+DATOS+'_nulos.csv',columns=['Tweet','etiqueta'],index=False)


nonulos=lista[(lista['et_spl']!=0.0) | (lista['et_senticon']!=0.0) | (lista['et_isol']!=0.0) | (lista['et_crisol']!=0.0)]
nonulos=nonulos.reset_index(drop=True)
nonulos.to_csv(PATH+DATOS+'_nonulos.csv',columns=['Tweet','etiqueta'],index=False)

#Eliminación de los neutros, solo queremos polaridad
t0=time.time()
lista=lista[lista.etiqueta!=0.0]
lista=lista.reset_index(drop=True)
time_elim=time.time()-t0
print("\nTiempo eliminación neutros:",round(time_elim,2),"seg")
lista.to_csv(PATH+DATOS+'_etiq.csv',columns=['Tweet','etiqueta'],index=False)

print("\n-----ARCHIVOS GENERADOS-----\n")
print("Datos etiquetados (con los neutros): ..."+DATOS+"_etiq_neutro.csv")
print("Datos etiquetados (sin los neutros): ..."+DATOS+"_etiq.csv")
print("Tweets no nulos: ..."+DATOS+"_nonulos.csv")
print("Tweets nulos: ..."+DATOS+"_nulos.csv")
