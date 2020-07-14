#Autor: Alberto Calvo Madurga

import json
import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import time
import string
import nltk
import sys
from nltk.tokenize import word_tokenize
import warnings;warnings.filterwarnings("ignore")
#preprocesamiento
import preprocessing_data as pre
#Lemmatizador
import es_core_news_sm
nlp=es_core_news_sm.load()

####### PARÁMETROS Y ARGUMENTOS ########
PATH="C:/Users/ASUS/Documents/tfg/tfg-acm-emorec/project/"
if len(sys.argv)==2:
    DATOS=sys.argv[1]
    DATOS=DATOS[:-4]
    PATH_IMAGES="C:/Users/ASUS/Documents/tfg/tfg-acm-emorec/project/datos_nuevos/"
    # Lectura de los datos
    lista = pd.read_csv(PATH+DATOS+"_etiq_neutro.csv")
    lista_nulos = pd.read_csv(PATH+DATOS+"_nulos.csv")
    lista_nonulos = pd.read_csv(PATH+DATOS+"_nonulos.csv")
else:
    DATOS="data/tweet15k_etiq_neutro.csv"
    PATH_IMAGES="C:/Users/ASUS/Documents/tfg/tfg-acm-emorec/project/data/"
    # Lectura de los datos
    lista = pd.read_csv(PATH+DATOS)
    lista_nulos = pd.read_csv(PATH+"data/tweet15k_nulos.csv")
    lista_nonulos = pd.read_csv(PATH+"data/tweet15k_nonulos.csv")



##############################################################################
t0=time.time()
# Preprocesamiento de los tweets
preprocessed_tweets=[]
for i in range(0,len(lista)):
    preprocessed_tweets.append(pre.preprocessing_tweet_es(lista.Tweet[i]))

preprocessed_nulos=[]
for i in range(0,len(lista_nulos)):
    preprocessed_nulos.append(pre.preprocessing_tweet_es(lista_nulos.Tweet[i]))

preprocessed_nonulos=[]
for i in range(0,len(lista_nonulos)):
    preprocessed_nonulos.append(pre.preprocessing_tweet_es(lista_nonulos.Tweet[i]))

print("\n--------TAMAÑO CONJUNTOS DE DATOS-------\n")
print("Conjunto de tweets completo:",len(preprocessed_tweets))
print("Conjunto de tweets no nulos:",len(preprocessed_nonulos))
print("Conjunto de tweets nulos:",len(preprocessed_nulos))


#-------ANÁLISIS DE COMPOSICIÓN DE LOS TWEETS-------
print("\n--------ANÁLISIS DE COMPOSICIÓN DE LOS TWEETS--------\n")
# --- Análisis de longitudes

#    Total   #
total=[]
total_preproc=[]
for i in range(0,len(lista)):
    total.append(len(lista.Tweet[i].split()))
    total_preproc.append(len(preprocessed_tweets[i].split()))
print("------Conjunto Completo-----")
print("Número de elementos (sin preprocesamiento):",sum(total))
print("Número de elementos de media (sin preprocesamiento)",round(sum(total)/len(lista),2))
print("Número de elementos (con preprocesamiento):",sum(total_preproc))
print("Número de elementos de media (con preprocesamiento)",round(sum(total_preproc)/len(lista),2),"\n")

#    No Nulos   #
nonulos=[]
nonulos_preproc=[]
for i in range(0,len(lista_nonulos)):
    nonulos.append(len(lista_nonulos.Tweet[i].split()))
    nonulos_preproc.append(len(preprocessed_nonulos[i].split()))
print("------Conjunto Nonulos-----")
print("Número de elementos (sin preprocesamiento):",sum(nonulos))
print("Número de elementos de media (sin preprocesamiento)",round(sum(nonulos)/len(lista_nonulos),2))
print("Número de elementos (con preprocesamiento):",sum(nonulos_preproc))
print("Número de elementos de media (con preprocesamiento)",round(sum(nonulos_preproc)/len(lista_nonulos),2),"\n")

#    Nulos   #
nulos=[]
nulos_preproc=[]
for i in range(0,len(lista_nulos)):
    nulos.append(len(lista_nulos.Tweet[i].split()))
    nulos_preproc.append(len(preprocessed_nulos[i].split()))
print("------Conjunto Nulos-----")
print("Número de elementos (sin preprocesamiento):",sum(nulos))
print("Número de elementos de media (sin preprocesamiento)",round(sum(nulos)/len(lista_nulos),2))
print("Número de elementos (con preprocesamiento):",sum(nulos_preproc))
print("Número de elementos de media (con preprocesamiento)",round(sum(nulos_preproc)/len(lista_nulos),2),"\n")


# Boxplots con y sin preprocesamiento
data=[total,nulos,nonulos]
data_con=[total_preproc,nulos_preproc,nonulos_preproc]

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(14,4))
fig.suptitle('Longitudes de los tweets')
fig.subplots_adjust(wspace=0.5)
plt.setp((ax1, ax2),xticklabels=['Completo', 'Nulos', 'No Nulos'])
ax1.boxplot(data,showfliers=False)
ax1.set_title("Sin preprocesamiento")
ax1.set(ylabel="Número de palabras")
ax2.boxplot(data_con,showfliers=False)
ax2.set_title("Con preprocesamiento")
ax2.set(ylabel="Número de palabras")
fig.savefig(PATH_IMAGES+'analitics/boxplot_longitudes.png')

#-------ANÁLISIS DE EMOJIS-------
print("\n--------ANÁLISIS DE EMOJIS--------\n")
#Listado de emojis con su polaridad asociada  http://kt.ijs.si/data/Emoji_sentiment_ranking/
listaemojis=pd.read_csv(PATH+"/lexicos/emoji/Emoji Sentiment Ranking 1.0.csv")
listaemojis=listaemojis[listaemojis.Occurrences>5]
listaemojis.Negative=listaemojis.Negative/listaemojis.Occurrences
listaemojis.Positive=listaemojis.Positive/listaemojis.Occurrences
listaemojis.Neutral=listaemojis.Neutral/listaemojis.Occurrences

import emoji
# Emojis en lista completa
emojis=[]
nemoji=0 #Número de emojis en el corpus
nemoji_corpus=0  #Emojis que se encuentran en el léxico
numero_emojis=np.zeros(len(lista)) #Para contar número de tweets sin emojis
numero_emojis_corpus=np.zeros(len(lista)) #Para contar número de tweets sin emojis confrontado con el léxico
for i in range(0,len(lista)):
    emojis=pre.text_has_emoji(preprocessed_tweets[i])
    for e in emojis:
        nemoji+=1
        numero_emojis[i]+=1
        if listaemojis['Emoji'].eq(e).any():
            nemoji_corpus+=1
            numero_emojis_corpus[i]+=1

# Emojis en lista de nonulos
emojis_nonulos=[]
nemoji_nonulos=0 #Número de emojis en el corpus
nemoji_corpus_nonulos=0  #Emojis que se encuentran en el léxico
numero_emojis_nonulos=np.zeros(len(lista_nonulos)) #Para contar número de tweets sin emojis
numero_emojis_corpus_nonulos=np.zeros(len(lista_nonulos)) #Para contar número de tweets sin emojis confrontado con el léxico
for i in range(0,len(lista_nonulos)):
    emojis_nonulos=pre.text_has_emoji(preprocessed_nonulos[i])
    for e in emojis_nonulos:
        nemoji_nonulos+=1
        numero_emojis_nonulos[i]+=1
        if listaemojis['Emoji'].eq(e).any():
            nemoji_corpus_nonulos+=1
            numero_emojis_corpus_nonulos[i]+=1

# Emojis en lista de nulos
emojis_nulos=[]
nemoji_nulos=0 #Número de emojis en el corpus
nemoji_corpus_nulos=0  #Emojis que se encuentran en el léxico
numero_emojis_nulos=np.zeros(len(lista_nulos)) #Para contar número de tweets sin emojis
numero_emojis_corpus_nulos=np.zeros(len(lista_nulos)) #Para contar número de tweets sin emojis confrontado con el léxico
for i in range(0,len(lista_nulos)):
    emojis_nulos=pre.text_has_emoji(preprocessed_nulos[i])
    for e in emojis_nulos:
        nemoji_nulos+=1
        numero_emojis_nulos[i]+=1
        if listaemojis['Emoji'].eq(e).any():
            nemoji_corpus_nulos+=1
            numero_emojis_corpus_nulos[i]+=1

noemojis=numero_emojis[numero_emojis<1].size
noemojis_corpus=numero_emojis_corpus[numero_emojis_corpus<1].size

noemojis_nonulos=numero_emojis_nonulos[numero_emojis_nonulos<1].size
noemojis_corpus_nonulos=numero_emojis_corpus_nonulos[numero_emojis_corpus_nonulos<1].size

noemojis_nulos=numero_emojis_nulos[numero_emojis_nulos<1].size
noemojis_corpus_nulos=numero_emojis_corpus_nulos[numero_emojis_corpus_nulos<1].size

# Conjunto completo
print("------Conjunto Completo-----")
print("Número de emojis encontrados:",sum(numero_emojis))
print("Número de emojis en el léxico:",nemoji_corpus, '(',round(nemoji_corpus/sum(numero_emojis)*100,2),"%)")
print("Número de tweets sin emojis:",noemojis,"(",round(noemojis/len(lista)*100,2),"%)")
print("Media de emojis en tweets con emojis:",round(sum(numero_emojis)/(len(lista)-noemojis),2))
print("Número de tweets sin emojis en el léxico:",noemojis_corpus,"(",round(noemojis_corpus/len(lista)*100,2),"%)")
print("Media de emojis en tweets con emojis del léxico:",sum(numero_emojis_corpus)/(len(lista)-noemojis_corpus),"\n")

# Conjunto nonulos
print("------Conjunto Nonulos-----")
print("Número de emojis encontrados:",sum(numero_emojis_nonulos))
print("Número de emojis en el léxico:",nemoji_corpus_nonulos, '(',round(nemoji_corpus_nonulos/sum(numero_emojis_nonulos)*100,2),"%)")
print("Número de tweets sin emojis:",noemojis_nonulos,"(",round(noemojis_nonulos/len(lista_nonulos)*100,2),"%)")
print("Media de emojis en tweets con emojis:",round(sum(numero_emojis_nonulos)/(len(lista_nonulos)-noemojis_nonulos),2))
print("Número de tweets sin emojis en el léxico:",noemojis_corpus_nonulos,"(",round(noemojis_corpus_nonulos/len(lista_nonulos)*100,2),"%)")
print("Media de emojis en tweets con emojis del léxico:",sum(numero_emojis_corpus_nonulos)/(len(lista_nonulos)-noemojis_corpus_nonulos),"\n")

# Conjunto nulos
print("------Conjunto Nulos-----")
print("Número de emojis encontrados:",sum(numero_emojis_nulos))
print("Número de emojis en el léxico:",nemoji_corpus_nulos, '(',round(nemoji_corpus_nulos/sum(numero_emojis_nulos)*100,2),"%)")
print("Número de tweets sin emojis:",noemojis_nulos,"(",round(noemojis_nulos/len(lista_nulos)*100,2),"%)")
print("Media de emojis en tweets con emojis:",round(sum(numero_emojis_nulos)/(len(lista_nulos)-noemojis_nulos),2))
print("Número de tweets sin emojis en el léxico:",noemojis_corpus_nulos,"(",round(noemojis_corpus_nulos/len(lista_nulos)*100,2),"%)")
print("Media de emojis en tweets con emojis del léxico:",sum(numero_emojis_corpus_nulos)/(len(lista_nulos)-noemojis_corpus_nulos),"\n")

#-- Diagrama de barras de emojios/tweets
# Para todos los emojis
bar_data=np.zeros(10,dtype=int)
for i in range(1,10):
    bar_data[i-1]=len(numero_emojis[numero_emojis==i])
bar_data[9]=len(numero_emojis[numero_emojis>=10])

# Para los emojis que han confrontado
bar_data_corpus=np.zeros(10,dtype=int)
for i in range(1,10):
    bar_data_corpus[i-1]=len(numero_emojis_corpus[numero_emojis_corpus==i])
bar_data_corpus[9]=len(numero_emojis_corpus[numero_emojis_corpus>=10])

#Gráfico emojis total
bars=['1','2','3','4','5','6','7','8','9','10+']
freq=list(bar_data)
df = pd.DataFrame({"Palabras":bars,
                  "Frecuencia":freq})
df=df[0:10]
plt.figure(figsize=(10,6))
plt.bar('Palabras', 'Frecuencia',data=df,color="green")
plt.title("Distribución de emojis en tweets", size=18)
plt.ylabel("Nºtweets")
plt.xlabel("Nºemojis")
plt.savefig(PATH_IMAGES+'/analitics/freq_emojis_tweet.png')

# Gráfico emojis en léxico
bars_corpus=['1','2','3','4','5','6','7','8','9','10+']
freq_corpus=list(bar_data_corpus)
df_corpus = pd.DataFrame({"Palabras":bars_corpus,
                  "Frecuencia":freq_corpus})

df_corpus=df_corpus[0:10]
plt.figure(figsize=(10,6))
plt.bar('Palabras', 'Frecuencia',data=df_corpus,color="green")
plt.title("Distribución de emojis del léxico en tweets", size=18)
plt.ylabel("Nºtweets")
plt.xlabel("Nºemojis")
plt.savefig(PATH_IMAGES+'/analitics/freq_emojis_tweet_lexico.png')

# Gráfico comparativa emojis
plt.figure(figsize=(8,5))
plt.plot(bars, freq, marker = '*',label="Todos los emojis")
plt.plot(bars_corpus, freq_corpus, marker = '*',label="Solo en el léxico")
plt.title('Comparativa distribución de emojis en tweets')
plt.legend()
plt.ylabel("Nºtweets")
plt.xlabel("Nºemojis")
plt.savefig(PATH_IMAGES+'/analitics/comparativa_freq_emojis_tweet.png')

#-------ANÁLISIS DE TÉRMINOS (RAE)-------
print("\n--------ANÁLISIS DE TÉRMINOS (RAE)--------\n")

import csv
formas=[]
with open(PATH+'/lexicos/rae/10000_formas.txt', newline = '') as file:
        lector = csv.reader(file, delimiter='\t')
        for forma in lector:
            formas.append(forma)
formas=formas[1:]
formas_df=pd.DataFrame.from_records(formas,columns=["Nada","Forma","Frec_Abs","Freq_Rel"])
formas_df.drop(columns="Nada",inplace=True)
formas_df.Forma=formas_df.Forma.str.rstrip() #10000

formas_grande=[]
with open(PATH+'/lexicos/rae/CREA_total.txt', newline = '') as file:
        lector = csv.reader(file, delimiter='\t')
        for forma in lector:
            formas_grande.append(forma)
formas_grande=formas_grande[1:]
formas_grande=pd.DataFrame.from_records(formas_grande,columns=["Nada","Forma","Frec_Abs","Freq_Rel"])
formas_grande.drop(columns="Nada",inplace=True)
formas_grande.Forma=formas_grande.Forma.str.rstrip()
formas_grande.Frec_Abs=formas_grande.Frec_Abs.str.replace(',', '')
formas_grande.Frec_Abs=formas_grande.Frec_Abs.astype(float)
formas_grande=formas_grande[formas_grande.Frec_Abs>50] # 77729

#  Corpus con 10000 formas
print("----Corpus 10000 formas----\n")
#Formas en tweets completo
formas_rae_completo=0
for num in range(0,len(lista)):
    for palabra in [word for word in word_tokenize(preprocessed_tweets[num])]:
        if formas_df['Forma'].eq(palabra).any():
            formas_rae_completo+=1
print("Formas RAE completo:",formas_rae_completo)
print("Porcentaje:",formas_rae_completo/(sum(total_preproc)-sum(numero_emojis)))

#Formas en tweets no nulos
formas_rae_nonulo=0
for num in range(0,len(lista_nonulos)):
    for palabra in [word for word in word_tokenize(preprocessed_nonulos[num])]:
        if formas_df['Forma'].eq(palabra).any():
            formas_rae_nonulo+=1
print("Formas RAE nonulo:",formas_rae_nonulo)
print("Porcentaje:",formas_rae_nonulo/(sum(nonulos_preproc)-sum(numero_emojis_nonulos)))

#Formas en tweets nulos
formas_rae_nulo=0
for num in range(0,len(lista_nulos)):
    for palabra in [word for word in word_tokenize(preprocessed_nulos[num])]:
        if formas_df['Forma'].eq(palabra).any():
            formas_rae_nulo+=1
print("Formas RAE nulo:",formas_rae_nulo)
print("Porcentaje:",formas_rae_nulo/(sum(nulos_preproc)-sum(numero_emojis_nulos)),"\n")

#  Corpus con 77729 formas
print("----Corpus 77729 formas----\n")
#Formas en tweets completo
formas_rae_completo=0
for num in range(0,len(lista)):
    for palabra in [word for word in word_tokenize(preprocessed_tweets[num])]:
        if formas_grande['Forma'].eq(palabra).any():
            formas_rae_completo+=1
print("Formas RAE completo:",formas_rae_completo)
print("Porcentaje:",formas_rae_completo/(sum(total_preproc)-sum(numero_emojis)))

#Formas en tweets no nulos
formas_rae_nonulo=0
for num in range(0,len(lista_nonulos)):
    for palabra in [word for word in word_tokenize(preprocessed_nonulos[num])]:
        if formas_grande['Forma'].eq(palabra).any():
            formas_rae_nonulo+=1
print("Formas RAE nonulo:",formas_rae_nonulo)
print("Porcentaje:",formas_rae_nonulo/(sum(nonulos_preproc)-sum(numero_emojis_nonulos)))

#Formas en tweets nulos
formas_rae_nulo=0
for num in range(0,len(lista_nulos)):
    for palabra in [word for word in word_tokenize(preprocessed_nulos[num])]:
        if formas_grande['Forma'].eq(palabra).any():
            formas_rae_nulo+=1
print("Formas RAE nulo:",formas_rae_nulo)
print("Porcentaje:",formas_rae_nulo/(sum(nulos_preproc)-sum(numero_emojis_nulos)),"\n")

print("Tiempo de ejecución:",round(time.time()-t0,2),"seg")
