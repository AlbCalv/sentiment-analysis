# Análisis de sentimientos y emociones en redes sociales utilizando ML

Este proyecto está creado para la realización del Trabajo de Fin de Grado del doble Grado de Ingeniería Informática y Estadística en la Universidad de Valladolid bajo la tutela de Valentín Cardeñoso Payo.

Los archivos asociados a la construcción del sistema final se encuentran todos dentro de la carpeta ***./project*** y lo correspondiente al seguimiento del proyecto (cuaderno de bitácora y planning) se encuentra en la carpetea *./seguimiento*

## Trabajo realizado
--------------------

Recalcar que todas las pruebas se han realizado en Jupyter Notebook y posteriormente se han convertido en módulos distinguidos en programas Python normales.

Para la primera fase de extracción de los datos se ha utilizado **./project/twitter_scrapper.py** . Se obtiene un conjunto de datos que contiene 15000 tweets en castellano que incluyen la palabra clave ''Covid'' y que se encuentran en *./project/data/es/tweet15k.csv*

En el archivo **./project/etiquetado-datos.ipynb** se encuentra la construcción del sistema de etiquetado de los 15000 tweets que previamente habíamos extraido. Este sistema está basado en léxicos que a continuación se describen:
* **Sentiment Polarity Lexicons (SPL)**: Consta de dos listados de términos positivos y negativos. Se encuentra en *./project/data/lexicon/negative_words_es* y *./project/data/lexicon/positive_words_es* 
* **ML-SentiCon**: Se adapta la estructura del archivo inicialmente en xml para conseguir un listado de términos con su grado de polaridad asociado siendo este un valor entre -1  y 1 (-1 para más negativo y 1 para más positivo). Se encuentra en *./project/data/mlsenticon/senticon.xlsx*
* **iSOL**: Contiene un listado de términos etiquetados como positivo o negativo. Se encuentra en *./project/data/crisol/isol.xlsx*
* **CRiSOL**: Contiene un listado de términos con tres valores entre 0 y 1 asociados que se corresponden con la correspondencia a términos 'positivos', 'negativos' y 'neutro'. Se encuentra en *./project/data/crisol/crisol.xlsx*
* **Emoji Sentiment Ranking 1.0**: Contiene una lista de emojis con información asociada de la que nos quedaremos la proporción de veces que han aparecido en textos etiquetados como 'positivos' y 'negativos'. Se encuentra en *./project/data/emoji/Emoji Sentiment Ranking 1.0.csv*

Se han construido cuatro sistemas de etiquetado cada uno utilizando uno de los léxicos de términos y el diccionario de emojis. Después se han combinado los resultados de ambos para construir la clasificación final de los tweets.
Durante todo el proceso de etiquetado se ha mantenido la posibilidad de etiquetar un tweet como 'neutro' ante la ausencia de elementos que nos indiquen una cierta polaridad, sin embargo en el sistema final se ha reducido de 15000 a **10573** que son el número de tweets etiquetados como 'positivo' (**6493**) y 'negativo' (**4080**) únicamente.
Todo este proceso se ha realizado con el sistema de etiquetado incluido en **./project/labeling_data.py**

Finalmente trás una extensa tarea de selección de parámetros, tenemos un clasificador final construido como un sistema de votación de mayoría de tres clasificadores individuales:SVM, Random Forest y Regresión Logística. Como sistema de extracción de características se ha utilizado BoW. Todo esto lo encontramos en **./project/modelado.py**

Todo lo realizado en el trabajo de fin de grado ha sido partiendo del archivo **./project/data/tweets15k.csv** y se han ido generando distintos archivos de datos almacenados en **.project/data/** así como gráficos, modelos y resultados.


## Replicar el trabajo
----------------------
Se deja preparada la estructura de archivos para que el lector pueda disponer de el proyecto para replicar el trabajo desde su fase inicial de extraer datos. 

Para ello vale con ejecutar el programa **.project/twitter_scrapper.py** pasándole como argumento el número de tweets que se pretenden extraer. El archivo de datos generado estará situado en la carpeta **./project/datos_nuevos**, donde va a ir todos los archivos generados en este nuevo experimento.

Un ejemplo del órden de ejecución para un experimento que se realiza entero con 4000 tweets una vez clonado el proyecto, sería el siguiente:

```console

    python twitter_scrapper.py 4000
    
    python labeling_data.py datos_nuevos/tweets4000.csv > datos_nuevos/resultados_etiquetado.txt
    
    python corpus_analitics.py datos_nuevos/tweets4000.csv  > datos_nuevos/resultados_analisis_corpus.txt
    
    python modelado.py datos_nuevos/tweets4000_etiq.csv  > datos_nuevos/resultados_clasificación.txt

```
Las órdenes se corresponden a la CMD de Windows con Python 3 instalado como variable de sistema.


Copyright © 2020 Alberto Calvo

under GNU GLP v3 license