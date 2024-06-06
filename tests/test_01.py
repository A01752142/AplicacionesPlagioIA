# Autores: A01752142 - Sandra Ximena Téllez Olvera
#          A01749164 - Jeovani Hernandez Bastida
#          A01025261 - Maximiliano Carrasco Rojas

import os
import tempfile
import unittest
from preprocesamientoIA import docs_originales, docs_sospechosos, caracteristicas_textos, train_model, calculate_metrics, matriz_confusion, roc_curve, detectar_plagio

class TestPreprocesamiento(unittest.TestCase):
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
        Configura archivos temporales para las pruebas.
        
        Esta función se ejecuta antes de cada prueba. Crea directorios temporales y archivos de texto
        para los documentos originales y sospechosos que se utilizarán en las pruebas.
        """
        # Configurar archivos temporales para las pruebas
        self.original_dir = tempfile.TemporaryDirectory()
        self.suspicious_dir = tempfile.TemporaryDirectory()
        
        # Crear archivos originales
        self.original_files = {
            'doc1.txt': 'Este es el primer documento original.',
            'doc2.txt': 'Este es el segundo documento original.'
        }
        for filename, content in self.original_files.items():
            with open(os.path.join(self.original_dir.name, filename), 'w', encoding='latin-1') as f:
                f.write(content)
        
        # Crear archivos sospechosos
        self.suspicious_files = {
            'doc3.txt': 'Este es el primer documento sospechoso.',
            'doc4.txt': 'Este es el segundo documento sospechoso.'
        }
        for filename, content in self.suspicious_files.items():
            with open(os.path.join(self.suspicious_dir.name, filename), 'w', encoding='latin-1') as f:
                f.write(content)

    def tearDown(self):
        """
        Limpia los archivos temporales creados para las pruebas.
        
        Esta función se ejecuta después de cada prueba para eliminar los directorios y archivos temporales.
        """
        # Limpiar archivos temporales
        self.original_dir.cleanup()
        self.suspicious_dir.cleanup()

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
        resultado = docs_sospechosos(self.suspicious_dir.name)
        esperado = [(filename, content) for filename, content in self.suspicious_files.items()]
        self.assertEqual(resultado, esperado)

    def test_caracteristicas_textos(self):
        """
        Prueba la función caracteristicas_textos.
        
        Verifica que la función extrae correctamente las características de los textos proporcionados.
        """
        texts = [content for _, content in self.original_files.items()] + [content for _, content in self.suspicious_files.items()]
        caracteristicas = caracteristicas_textos(texts)
        self.assertEqual(caracteristicas.shape[0], len(texts))

    def test_train_model(self):
        """
        Prueba la función train_model.
        
        Verifica que la función entrena correctamente un modelo de clasificación para detectar plagio.
        """
        texts = [content for _, content in self.original_files.items()] + [content for _, content in self.suspicious_files.items()]
        caracteristicas = caracteristicas_textos(texts)
        num_originals = len(self.original_files)
        caracteristicas_originales = caracteristicas[:num_originals]
        caracteristicas_sospechosos = caracteristicas[num_originals:]
        
        model, y_test, y_pred, y_puntajes = train_model(caracteristicas_originales, caracteristicas_sospechosos)
        self.assertEqual(len(y_test), len(y_pred))
        self.assertEqual(len(y_test), len(y_puntajes))

    def test_calculate_metrics(self):
        """
        Prueba la función calculate_metrics.
        
        Verifica que la función calcula correctamente las métricas de rendimiento del modelo.
        """
        y_test = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 0]
        y_puntajes = [0.1, 0.9, 0.2, 0.4]
        auc, recall, f1 = calculate_metrics(y_test, y_pred, y_puntajes)
        self.assertTrue(0 <= auc <= 1)
        self.assertTrue(0 <= recall <= 1)
        self.assertTrue(0 <= f1 <= 1)

    def test_detectar_plagio(self):
        """
        Prueba la función detectar_plagio.
        
        Verifica que la función utiliza el modelo entrenado para detectar plagio en los textos sospechosos.
        """
        texts = [content for _, content in self.original_files.items()] + [content for _, content in self.suspicious_files.items()]
        caracteristicas = caracteristicas_textos(texts)
        num_originals = len(self.original_files)
        caracteristicas_originales = caracteristicas[:num_originals]
        caracteristicas_sospechosos = caracteristicas[num_originals:]
        
        model, y_test, y_pred, y_puntajes = train_model(caracteristicas_originales, caracteristicas_sospechosos)
        predicciones = detectar_plagio(model, caracteristicas_sospechosos)
        self.assertEqual(predicciones.shape[0], caracteristicas_sospechosos.shape[0])

if __name__ == '__main__':
    unittest.main()