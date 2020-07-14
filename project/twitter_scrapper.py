#Autor: Alberto Calvo Madurga

import json
import csv
import tweepy
import re
import os
import sys
import pandas as  pd
import time
import string
PATH="C:/Users/ASUS/Documents/tfg/tfg-acm-emorec/project"
#Claves Twitter
import claves_twitter as tw
#--------ARGUMENTOS-------------------
if len(sys.argv)==2:
    NTWEETS=int(sys.argv[1])
else:
    NTWEETS=15000
#-------EXTRACCIÓN DE LOS DATOS-------

# Autenticación para acceder a la API de Twitter
auth = tweepy.OAuthHandler(tw.CONSUMER_KEY, tw.CONSUMER_SECRET)
auth.set_access_token(tw.ACCESS_TOKEN, tw.ACCESS_TOKEN_SECRET)

# Inicialización de la API de Twitter
api = tweepy.API(auth)

# Evitar la tasa de limitación
api.wait_on_rate_limit=True

# Creación del dataframe donde se almacenarán las búsquedas
tweets=pd.DataFrame({'Tweet':[],})
ind=0

# Recogida de tweets que coincidan con la palabra en q
t0=time.time()
for tweet in tweepy.Cursor(api.search, q='Covid -filter:retweets', lang="es", tweet_mode='extended').items(NTWEETS):
    texto=tweet.full_text.replace('\n',' ')
    tweets.loc[ind]=[texto]
    ind+=1
print("Tiempo de extracción para",NTWEETS,"tweets:",round(time.time()-t0,2),"seg")

# Exportación de los datos
lista=pd.DataFrame(tweets.Tweet)
lista.to_csv(str(PATH)+'/datos_nuevos/tweets'+str(NTWEETS)+'.csv',index=False)
print("\nArchivo generado: "+str(PATH)+'/datos_nuevos/tweets'+str(NTWEETS)+'.csv')
