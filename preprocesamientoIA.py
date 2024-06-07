# Autores: A01752142 - Sandra Ximena Téllez Olvera
#          A01749164 - Jeovani Hernandez Bastida
#          A01025261 - Maximiliano Carrasco Rojas

import os
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import scipy.sparse
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, f1_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from Utilities import preprocess_text
from tipo_plagio import tipo_plagio


def docs_originales(originales):
    """
    Carga los documentos originales desde el directorio especificado.
    Parameters:
        originales (str): Ruta del directorio que contiene los documentos originales.
    Returns:
        list: Lista de tuplas (nombre_archivo, contenido) de los documentos originales.
    """
    originals = []
    for filename in os.listdir(originales):
        with open(os.path.join(originales, filename), 'r', encoding='latin-1') as file:
            text = file.read()
            originals.append((filename, text))
    return originals    

def docs_sospechosos(sospechosos):
    """
    Carga los documentos sospechosos desde el directorio especificado.
    Parameters:
        sospechosos (str): Ruta del directorio que contiene los documentos sospechosos.
    Returns:
        list: Lista de tuplas (nombre_archivo, contenido) de los documentos sospechosos.
    """
    suspicious = []
    for filename in os.listdir(sospechosos):
        with open(os.path.join(sospechosos, filename), 'r', encoding='latin-1') as file:
            text = file.read()
            suspicious.append((filename, text))
    return suspicious

def caracteristicas_textos(texts):
    """
    Extrae características TF-IDF de los textos dados.
    Parameters:
        texts (list): Lista de textos a procesar.
    Returns:
        scipy.sparse.csr_matrix: Matriz de características TF-IDF.
    """
    vectorizar_textos = TfidfVectorizer()
    caracteristicas = vectorizar_textos.fit_transform(texts)
    return caracteristicas

def train_model(caracteristicas_originales, caracteristicas_sospechosos):
    """
    Entrena un modelo SVM utilizando las características TF-IDF.
    Parameters:
        caracteristicas_originales (scipy.sparse.csr_matrix): Matriz de características de los documentos originales.
        caracteristicas_sospechosos (scipy.sparse.csr_matrix): Matriz de características de los documentos sospechosos.
    Returns:
        tuple: Modelo entrenado, etiquetas de prueba, predicciones y puntajes de decisión.
    """
    matriz_caracteristicas = scipy.sparse.vstack([caracteristicas_originales, caracteristicas_sospechosos])
    num_originals = caracteristicas_originales.shape[0]
    num_copies = caracteristicas_sospechosos.shape[0]
    etiquetas = [0] * num_originals + [1] * num_copies
    X_train, X_test, y_train, y_test = train_test_split(matriz_caracteristicas, etiquetas, test_size=0.26, random_state=23)  #se modifico .25 a .4
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_puntajes = model.decision_function(X_test)  # Get decision scores for ROC
    return model, y_test, y_pred, y_puntajes

def calculate_metrics(y_test, y_pred, y_puntajes):
    """
    Calcula métricas de rendimiento del modelo.
    Parameters:
        y_test (list): Etiquetas reales.
        y_pred (list): Predicciones del modelo.
        y_puntajes (list): Puntajes de decisión del modelo.
    Returns:
        tuple: AUC, Recall y F1 Score.
    """
    auc = roc_auc_score(y_test, y_puntajes)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return auc, recall, f1

def matriz_confusion(y_test, y_pred):
    """
    Muestra la matriz de confusión de las predicciones del modelo.
    Parameters:
        y_test (list): Etiquetas reales.
        y_pred (list): Predicciones del modelo.
    """
    matriz = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ax.matshow(matriz, cmap=plt.cm.Blues)
    for (i, j), val in np.ndenumerate(matriz):
        ax.text(j, i, f'{val}', ha='center', va='center', color='red')
    plt.xlabel('Predicción')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def roc_curve(y_test, y_puntajes):
    """
    Muestra la curva ROC de las predicciones del modelo.
    Parameters:
        y_test (list): Etiquetas reales.
        y_puntajes (list): Puntajes de decisión del modelo.
    """
    RocCurveDisplay.from_predictions(y_test, y_puntajes)
    plt.title('ROC Curve')
    plt.show()

def detectar_plagio(model, caracteristicas_sospechosos):
    """
    Detecta plagio en los documentos sospechosos utilizando el modelo entrenado.
    Parameters:
        model (SVC): Modelo entrenado.
        caracteristicas_sospechosos (scipy.sparse.csr_matrix): Matriz de características de los documentos sospechosos.
    Returns:
        list: Predicciones del modelo para los documentos sospechosos.
    """
    predicciones = model.predict(caracteristicas_sospechosos)
    return predicciones

def main():
    """
    Función principal que ejecuta las operaciones del módulo, incluyendo carga de datos, procesamiento,
    entrenamiento del modelo, evaluación y detección de plagio.
    """
    originales = "Documentos"
    sospechosos = "Documentos_comparar"
    originals = docs_originales(originales)
    suspicious = docs_sospechosos(sospechosos)
    all_documents = originals + suspicious
    preprocessed_documents = [preprocess_text(text) for _, text in all_documents]
    caracteristicas = caracteristicas_textos(preprocessed_documents)
    num_originals = len(originals)
    caracteristicas_originales = caracteristicas[:num_originals]
    caracteristicas_sospechosos = caracteristicas[num_originals:]
    
    # Entrenando el modelo y obteniendo predicciones
    start_time = time.time()
    model, y_test, y_pred, y_puntajes = train_model(caracteristicas_originales, caracteristicas_sospechosos)
    
    # Calcular métricas
    auc, recall, f1 = calculate_metrics(y_test, y_pred, y_puntajes)   
    print(f"AUC: {auc}, Recall: {recall}, F1 Score: {f1}")
    
    # Mostrar matriz de confusión y curva ROC
    matriz_confusion(y_test, y_pred)
    roc_curve(y_test, y_puntajes)
    
    end_time = time.time()
    print(f"Tiempo de ejecución: {end_time - start_time} segundos")
    
    # Detección de plagio individual y escribir resultados
    predicciones = detectar_plagio(model, caracteristicas_sospechosos)
    plagiarism_results = []
    df = pd.DataFrame(columns=['nombre_original', 'texto_original', 'nombre_sospechoso', 'texto_sospechoso', 'is_copy', 'copy_type', 'percentage'])
    for i, copy_doc in enumerate(suspicious):
        nombre_sospechoso, texto_sospechoso = copy_doc
        plagio_resultado_copia = []
        for j, (nombre_original, texto_original) in enumerate(originals):
            tipo, porcentaje = tipo_plagio(texto_original, texto_sospechoso)
            plagio_encontrado = predicciones[i] == 1
            if not plagio_encontrado:
                tipo = 'None'
                porcentaje = 0.0
            plagio_resultado_copia.append((nombre_original, plagio_encontrado, tipo, porcentaje))
            df = pd.concat([df, pd.DataFrame({'nombre_original': [nombre_original],
                                            'texto_original': [texto_original],
                                            'nombre_sospechoso': [nombre_sospechoso],
                                            'texto_sospechoso': [texto_sospechoso],
                                            'is_sospechoso': [plagio_encontrado],
                                            'tipo': [tipo],
                                            'porcentaje': [porcentaje]})])
            
        # Ordenar los resultados por porcentaje de similitud y seleccionar los cinco primeros
        plagio_resultado_copia.sort(key=lambda x: x[3], reverse=True)
        top_5_results = plagio_resultado_copia[:2]
        plagiarism_results.append((nombre_sospechoso, top_5_results))
        temp_df = pd.DataFrame({
        'nombre_original': [res[0] for res in top_5_results],
        'texto_original': [originals[j][1] for j in range(len(top_5_results))],
        'nombre_sospechoso': [nombre_sospechoso] * len(top_5_results),
        'texto_sospechoso': [texto_sospechoso] * len(top_5_results),
        'is_sospechoso': [res[1] for res in top_5_results],
        'tipo': [res[2] for res in top_5_results],
        'porcentaje': [res[3] for res in top_5_results]
    })

    # Ignora el índice para evitar el FutureWarning
    df = pd.concat([df, temp_df], ignore_index=True)  
    
    # Escribir los resultados en un archivo de texto
    with open('Resultados/plagio_resultado.txt', 'w') as file:
        for result in plagiarism_results:
            nombre_sospechoso, top_5_results = result
            file.write(f'\n************* Archivo copia: {nombre_sospechoso} **************\n')
            for nombre_original, plagio_encontrado, tipo, porcentaje in top_5_results:
                file.write(f'Archivo original: {nombre_original}\n')
                file.write(f"¿Es plagio?: {'Si' if plagio_encontrado else 'No'}\n")
                file.write(f"Tipo de plagio: {'Ninguno' if plagio_encontrado == 'None' else tipo}\n")
                file.write(f"Porcentaje de plagio: {round(porcentaje, 2)}%\n")
                file.write("------------------------------------------\n")
                
    # Guardar los resultados en un archivo Excel
    df.to_excel('Resultados/plagio_resultado.xlsx', index=False)

if __name__ == "__main__":
    main()