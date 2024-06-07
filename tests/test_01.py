import unittest
import os
import tempfile
from unittest.mock import patch
import scipy.sparse
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, recall_score, f1_score
import pandas as pd
import numpy as np

from preprocesamientoIA import (
    docs_originales,
    docs_sospechosos,
    caracteristicas_textos,
    train_model,
    calculate_metrics
)

class TestPlagiarismDetection(unittest.TestCase):
    """
    Clase de pruebas unitarias para las funciones del módulo preprocesamiento.
    
    Esta clase realiza pruebas sobre las siguientes funciones: 
    - docs_originales: Carga los documentos originales desde un directorio dado.
    - docs_sospechosos: Carga los documentos sospechosos desde un directorio dado.
    - caracteristicas_textos: Extrae características de los textos proporcionados.
    - train_model: Entrena un modelo de clasificación para detectar plagio.
    - calculate_metrics: Calcula las métricas de rendimiento del modelo.
    - detectar_plagio: Utiliza el modelo entrenado para detectar plagio en textos sospechosos.
    """
    
    def setUp(self):
        """
        Configuración de archivos temporales y directorios para pruebas.
        """
        # Crear directorio temporal para documentos originales
        self.original_dir = tempfile.TemporaryDirectory()
        self.original_files = {
            "doc1.txt": "Este es un documento original.",
            "doc2.txt": "Este es otro documento original."
        }
        for filename, content in self.original_files.items():
            with open(os.path.join(self.original_dir.name, filename), 'w', encoding='latin-1') as f:
                f.write(content)

        # Crear directorio temporal para documentos sospechosos
        self.sospechoso_dir = tempfile.TemporaryDirectory()
        self.sospechoso_files = {
            "sus1.txt": "Este es un documento sospechoso.",
            "sus2.txt": "Este es otro documento sospechoso."
        }
        for filename, content in self.sospechoso_files.items():
            with open(os.path.join(self.sospechoso_dir.name, filename), 'w', encoding='latin-1') as f:
                f.write(content)

    def tearDown(self):
        """
        Limpieza de archivos temporales y directorios después de las pruebas.
        """
        self.original_dir.cleanup()
        self.sospechoso_dir.cleanup()

    def test_docs_originales(self):
        """
        Prueba la función docs_originales.
        Verifica que la función carga correctamente los documentos originales desde el directorio especificado.
        """
        resultado = docs_originales(self.original_dir.name)
        esperado = [(filename, content) for filename, content in self.original_files.items()]
        self.assertEqual(resultado, esperado)

    def test_docs_sospechosos(self):
        """
        Prueba la función docs_sospechosos.
        Verifica que la función carga correctamente los documentos sospechosos desde el directorio especificado.
        """
        resultado = docs_sospechosos(self.sospechoso_dir.name)
        esperado = [(filename, content) for filename, content in self.sospechoso_files.items()]
        self.assertEqual(resultado, esperado)

    def test_caracteristicas_textos(self):
        """
        Prueba la función caracteristicas_textos.
        Verifica que la función extrae correctamente las características TF-IDF de los textos dados.
        """
        textos = ["Este es un texto.", "Este es otro texto."]
        caracteristicas = caracteristicas_textos(textos)
        self.assertIsInstance(caracteristicas, scipy.sparse.csr_matrix)
        self.assertEqual(caracteristicas.shape[0], len(textos))

@patch('preprocesamientoIA.SVC', spec=True)
def test_train_model(self, MockSVC):
    """
    Prueba la función train_model.
    Verifica que la función entrena correctamente un modelo SVM utilizando las características TF-IDF.
    """
    MockSVC.return_value.predict.return_value = [1, 0]
    MockSVC.return_value.decision_function.return_value = [0.6, -0.2]
    
    textos = ["Este es un texto.", "Este es otro texto."]
    caracteristicas = caracteristicas_textos(textos)
    caracteristicas_originales = caracteristicas[:1]
    caracteristicas_sospechosos = caracteristicas[1:]
    
    model, y_test, y_pred, y_puntajes = train_model(caracteristicas_originales, caracteristicas_sospechosos)
    
    self.assertIsInstance(model, SVC)  # Verifica que el modelo sea una instancia de SVC
    self.assertIsInstance(y_test, list)
    self.assertIsInstance(y_pred, np.ndarray)
    self.assertIsInstance(y_puntajes, np.ndarray)

    def test_calculate_metrics(self):
        """
        Prueba la función calculate_metrics.
        Verifica que la función calcula correctamente las métricas de rendimiento del modelo.
        """
        y_test = [0, 1]
        y_pred = [0, 1]
        y_puntajes = [0.2, 0.8]
        
        auc, recall, f1 = calculate_metrics(y_test, y_pred, y_puntajes)
        
        self.assertEqual(auc, roc_auc_score(y_test, y_puntajes))
        self.assertEqual(recall, recall_score(y_test, y_pred))
        self.assertEqual(f1, f1_score(y_test, y_pred))

if __name__ == '__main__':
    unittest.main()
