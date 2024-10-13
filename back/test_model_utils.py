import unittest
from model_utils import lemmatize_and_remove_stopwords

class TestModelUtils(unittest.TestCase):
    def test_lemmatize_and_remove_stopwords(self):
        text = "Este es un ejemplo de texto para probar el preprocesamiento."
        result = lemmatize_and_remove_stopwords(text)
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")

if __name__ == '__main__':
    unittest.main()
