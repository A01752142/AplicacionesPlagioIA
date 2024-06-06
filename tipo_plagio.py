# Autores: A01752142 - Sandra Ximena Téllez Olvera
#          A01749164 - Jeovani Hernandez Bastida
#          A01025261 - Maximiliano Carrasco Rojas

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from Utilities import preprocess_text, similitud_coseno

def plagio_parafraseo(original_text, suspicious_text):
    """
    Detecta plagio por parafraseo entre dos textos.
    Parameters:
        original_text (str): Texto original.
        suspicious_text (str): Texto sospechoso de plagio.
    Returns:
        tuple: Tipo de plagio ("Paraphrasing") y porcentaje de similitud si es mayor o igual al 30%, 
               de lo contrario None y 0.
    """
    processed_original = preprocess_text(original_text)
    processed_suspicious = preprocess_text(suspicious_text)
    similarity = similitud_coseno(processed_original, processed_suspicious)
    paraphrase_percentage = similarity * 100
    if paraphrase_percentage >= 30:
        return "Parafraseo", paraphrase_percentage
    return None, 0

def plagio_desordenar_frases(original_text, suspicious_text):
    """
    Detecta plagio por desordenar frases entre dos textos.
    Parameters:
        original_text (str): Texto original.
        suspicious_text (str): Texto sospechoso de plagio.
    Returns:
        tuple: Tipo de plagio ("Sentence Shuffling") y porcentaje de similitud.
    """
    original_sentences = nltk.sent_tokenize(original_text)
    suspicious_sentences = nltk.sent_tokenize(suspicious_text)
    vectorizer = TfidfVectorizer()
    text_matrix = vectorizer.fit_transform(original_sentences + suspicious_sentences)
    similarity = cosine_similarity(text_matrix[0], text_matrix[1])[0][0]
    return "Desordenar frases", similarity * 100

def plagio_cambio_tiempo(original_text, suspicious_text):
    """
    Detecta plagio por cambio de tiempo verbal entre dos textos.
    Parameters:
        original_text (str): Texto original.
        suspicious_text (str): Texto sospechoso de plagio.
    Returns:
        tuple: Tipo de plagio ("Tense Change (Past)" o "Tense Change (Present)") y el mayor porcentaje de palabras 
               en tiempo pasado o presente si la diferencia es mayor a 15%, de lo contrario None y 0.
    """
    past_tense_words = {'was', 'were', 'had', 'did'}
    present_tense_words = {'is', 'are', 'has', 'does'}
    original_tokens = set(word_tokenize(original_text.lower()))
    suspicious_tokens = set(word_tokenize(suspicious_text.lower()))
    past_tense_original = sum(word in past_tense_words for word in original_tokens)
    past_tense_suspicious = sum(word in past_tense_words for word in suspicious_tokens)
    present_tense_original = sum(word in present_tense_words for word in original_tokens)
    present_tense_suspicious = sum(word in present_tense_words for word in suspicious_tokens)
    past_tense_percentage_original = past_tense_original / max(len(original_tokens), 1) * 100
    past_tense_percentage_suspicious = past_tense_suspicious / max(len(suspicious_tokens), 1) * 100
    present_tense_percentage_original = present_tense_original / max(len(original_tokens), 1) * 100
    present_tense_percentage_suspicious = present_tense_suspicious / max(len(suspicious_tokens), 1) * 100
    if abs(past_tense_percentage_original - past_tense_percentage_suspicious) > 15:
        return "Cambio de tiempo (pasado)", max(past_tense_percentage_original, past_tense_percentage_suspicious)
    if abs(present_tense_percentage_original - present_tense_percentage_suspicious) > 15:
        return "Cambio de tiempo (presente)", max(present_tense_percentage_original, present_tense_percentage_suspicious)
    return None, 0

def plagio_cambio_voz(original_text, suspicious_text):
    """
    Detecta plagio por cambio de voz (activa/pasiva) entre dos textos.
    Parameters:
        original_text (str): Texto original.
        suspicious_text (str): Texto sospechoso de plagio.
    Returns:
        tuple: Tipo de plagio ("Voice Change") y el mayor porcentaje de uso de pronombres si la diferencia es mayor a 20%, 
               de lo contrario None y 0.
    """
    pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they'}
    original_tokens = word_tokenize(original_text.lower())
    suspicious_tokens = word_tokenize(suspicious_text.lower())
    pronouns_original = sum(token in pronouns for token in original_tokens)
    pronouns_suspicious = sum(token in pronouns for token in suspicious_tokens)
    pronouns_percentage_original = pronouns_original / max(len(original_tokens), 1) * 100
    pronouns_percentage_suspicious = pronouns_suspicious / max(len(suspicious_tokens), 1) * 100
    if abs(pronouns_percentage_original - pronouns_percentage_suspicious) > 20:
        return "Cambio de voz", max(pronouns_percentage_original, pronouns_percentage_suspicious)
    return None, 0

def plagio_insertar_frases(original_text, suspicious_text):
    """
    Detecta plagio por inserción o reemplazo de frases entre dos textos.
    Parameters:
        original_text (str): Texto original.
        suspicious_text (str): Texto sospechoso de plagio.
    Returns:
        tuple: Tipo de plagio ("Inserción o reemplazo de frases") y el mayor porcentaje de palabras comunes si ambos
               porcentajes son mayores al 25%, de lo contrario None y 0.
    """
    original_tokens = set(word_tokenize(original_text.lower()))
    suspicious_tokens = set(word_tokenize(suspicious_text.lower()))
    common_tokens = original_tokens.intersection(suspicious_tokens)
    if len(original_tokens) == 0 or len(suspicious_tokens) == 0:
        return None, 0
    common_percentage_original = len(common_tokens) / len(original_tokens)
    common_percentage_suspicious = len(common_tokens) / len(suspicious_tokens)
    if common_percentage_original > 0.25 and common_percentage_suspicious > 0.25:
        return "Insertar o reemplazar frases", max(common_percentage_original, common_percentage_suspicious) * 100
    return None, 0

def tipo_plagio(original_text, suspicious_text):
    """
    Detecta el tipo de plagio entre dos textos, evaluando varios métodos.
    Parameters:
        original_text (str): Texto original.
        suspicious_text (str): Texto sospechoso de plagio.
    Returns:
        tuple: Tipo de plagio con el mayor porcentaje y el porcentaje correspondiente, 
               o "None" y 0 si no se detecta plagio.
    """
    types = []
    percentages = []
    results = [
        plagio_desordenar_frases(original_text, suspicious_text),
        plagio_cambio_tiempo(original_text, suspicious_text),
        plagio_cambio_voz(original_text, suspicious_text),
        plagio_insertar_frases(original_text, suspicious_text),
        plagio_parafraseo(original_text, suspicious_text)
    ]
    for result in results:
        if result[0]:
            types.append(result[0])
            percentages.append(result[1])
    if percentages:
        max_percentage_index = percentages.index(max(percentages))
        return types[max_percentage_index], percentages[max_percentage_index]
    return "None", 0