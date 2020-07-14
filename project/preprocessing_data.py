#Autor: Alberto Calvo Madurga

import pandas as pd
import numpy as np
import string
import emoji
import re
import nltk
import csv
import xlrd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#Lemmatizador
import es_core_news_sm
nlp=es_core_news_sm.load()
import warnings;warnings.filterwarnings("ignore")
PATH="C:/Users/ASUS/Documents/tfg/tfg-acm-emorec/project/data"

SIGNOS_PUNTUACION=string.punctuation+'¿¡'



# Funciones destinadas al preprocesamiento del texto
def preprocessing_tweet_es(text):
    # Eliminar las URLs
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text)
    text = re.sub(r'http\S+', '', text)
    # Eliminar saltos de línea y cosas variadas
    text = re.sub('\n','',text)
    # Eliminar las menciones (tener en cuenta el fallo clásico de primero arroba y luego espacio)
    text = re.sub('@\s','@',text)
    text = re.sub('(@[^\s]+)|(w/)', '', text)
    # Eliminación de Hashtags
    text = re.sub(r'#([^\s]+)', r'\1', text)
    # Tokenización eliminando los signos de puntuación
    text = [char for char in text if char not in SIGNOS_PUNTUACION]
    # Unir de nuevo las palabras
    text = ''.join(text)
    # Conversión a minúsculas
    text = text.lower()
    #Eliminar por la palabra buscada
    text=re.sub('covid19','',text)
    text=re.sub('covid 19','',text)
    text=re.sub('covid','',text)
    text=re.sub('coronavirus','',text)

    #Lematización
    lemma_tweet=[]
    for token in nlp(text):
        if str(token) not in stopwords.words('spanish'):
            lemma_tweet.append(token.lemma_)
    return ' '.join(lemma_tweet)

#Función que detecta emojis de las cadenas de texto
def text_has_emoji(text):
    lista=[]
    for character in text:
        if character in emoji.UNICODE_EMOJI:
            lista.append(character)
    return lista
