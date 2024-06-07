# Autores: A01752142 - Sandra Ximena Téllez Olvera
#          A01749164 - Jeovani Hernandez Bastida
#          A01025261 - Maximiliano Carrasco Rojas

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Descarga de datos necesarios para procesamiento de texto
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """
    Procesa el texto eliminando stopwords, aplicando stemming y lematización.
    Parameters:
        text (str): Texto a ser procesado.
    Returns:
        str: Texto procesado con tokens filtrados, stemizados y lematizados.
    """
    # Tokenización del texto
    tokens = word_tokenize(text)
    
    # Filtrado de stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Lematización
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    # Unión de tokens procesados en una cadena
    return ' '.join(lemmatized_tokens)

def similitud_coseno(texto1, texto2):
    """
    Calcula la similitud coseno entre dos textos.
    Parameters:
        texto1 (str): Primer texto.
        texto2 (str): Segundo texto.
    Returns:
        float: Valor de similitud coseno entre los textos.
    """
    
    # Vectorización TF-IDF de los textos
    vectorizar_textos = TfidfVectorizer()
    matriz_textos = vectorizar_textos.fit_transform([texto1, texto2])
    
    # Cálculo de la similitud coseno
    similitud = cosine_similarity(matriz_textos[0], matriz_textos[1])[0][0]
    return similitud