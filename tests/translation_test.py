import unittest

from src.translation import _detect_language, _translate


class TestDetectLanguage(unittest.TestCase):
    def test_detect_language_en(self):
        language = _detect_language("What is Duchenne Muscular Dystrophy?")
        self.assertEqual(language, "en")

    def test_detect_language_fr(self):
        language = _detect_language("Qu'est-ce que la dystrophie musculaire de Duchenne?")
        self.assertEqual(language, "fr")

    def test_detect_language_zh_CN(self):
        language = _detect_language("什么是杜氏肌营养不良症？")
        self.assertEqual(language, "zh-CN")

    def test_detect_language_zh_TW(self):
        language = _detect_language("什麼是杜氏肌肉營養不良症？")
        self.assertEqual(language, "zh-TW")


class TestTranslate(unittest.TestCase):
    def test_translate_fr_en(self):
        translation = _translate("Qu'est-ce que la dystrophie musculaire de Duchenne?", target="en")
        self.assertEqual(translation, "What is Duchenne muscular dystrophy?")

    def test_translate_zh_CN_en(self):
        translation = _translate("什么是杜氏肌营养不良症？", target="en")
        self.assertEqual(translation, "What is Duchenne muscular dystrophy?")

    def test_translate_zh_TW_en(self):
        translation = _translate("什麼是杜氏肌肉營養不良症？", target="en")
        self.assertEqual(translation, "What is Duchenne muscular dystrophy?")

    def test_translate_en_fr(self):
        translation = _translate("What is Duchenne Muscular Dystrophy?", target="fr")
        self.assertEqual(translation, "Qu’est-ce que la dystrophie musculaire de Duchenne ?")

    def test_translate_en_zh_CN(self):
        translation = _translate("What is Duchenne Muscular Dystrophy?", target="zh-CN")
        self.assertEqual(translation, "什么是杜氏肌营养不良症？")

    def test_translate_en_zh_TW(self):
        translation = _translate("What is Duchenne Muscular Dystrophy?", target="zh-TW")
        self.assertEqual(translation, "什麼是杜氏肌肉營養不良症？")
