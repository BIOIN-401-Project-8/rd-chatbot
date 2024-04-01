import pytest
from deep_translator import GoogleTranslator
from lingua import IsoCode639_1

from src.translation import _detect_language, _translate, get_language_detector
from translators.opusmt import OpusMTTranslator


@pytest.fixture
def detector():
    return get_language_detector(IsoCode639_1.EN, IsoCode639_1.FR, IsoCode639_1.IT, IsoCode639_1.ZH)


class TestDetectLanguage:
    def test_detect_language_en(self, detector):
        language = _detect_language(detector, "What is Duchenne Muscular Dystrophy?")["language"]
        assert language == "en"
        language = _detect_language(detector, "What treats phenylketonuria?")["language"]
        assert language == "en"

    def test_detect_language_fr(self, detector):
        language = _detect_language(detector, "Qu'est-ce que la dystrophie musculaire de Duchenne?")["language"]
        assert language == "fr"

    def test_detect_language_it(self, detector):
        language = _detect_language(detector, "Cos'è la distrofia muscolare di Duchenne?")["language"]
        assert language == "it"

    def test_detect_language_zh_CN(self, detector):
        language = _detect_language(detector, "什么是杜氏肌营养不良症？")["language"]
        assert language == "zh-CN"

    def test_detect_language_zh_TW(self, detector):
        language = _detect_language(detector, "什麼是杜氏肌肉營養不良症？")["language"]
        assert language == "zh-TW"


@pytest.fixture
def google_translator():
    return GoogleTranslator(source="auto", target="en")


@pytest.fixture
def opusmt_translator():
    return OpusMTTranslator(source="fr", target="en")


class TestGoogleTranslator:
    def test_translate_fr_en(self, google_translator):
        translation = _translate(
            google_translator, "Qu'est-ce que la dystrophie musculaire de Duchenne?", source="fr", target="en"
        )
        assert translation == "What is Duchenne muscular dystrophy?"

    def test_translate_it_en(self, google_translator):
        translation = _translate(
            google_translator, "Cos'è la distrofia muscolare di Duchenne?", source="it", target="en"
        )
        assert translation == "What is Duchenne muscular dystrophy?"

    def test_translate_zh_CN_en(self, google_translator):
        translation = _translate(google_translator, "什么是杜氏肌营养不良症？", source="zh-CN", target="en")
        assert translation == "What is Duchenne muscular dystrophy?"

    def test_translate_zh_TW_en(self, google_translator):
        translation = _translate(google_translator, "什麼是杜氏肌肉營養不良症？", source="zh-TW", target="en")
        assert translation == "What is Duchenne muscular dystrophy?"

    def test_translate_en_fr(self, google_translator):
        translation = _translate(google_translator, "What is Duchenne Muscular Dystrophy?", source="en", target="fr")
        assert translation == "Qu’est-ce que la dystrophie musculaire de Duchenne ?"

    def test_translate_en_it(self, google_translator):
        translation = _translate(google_translator, "What is Duchenne Muscular Dystrophy?", source="en", target="it")
        assert translation == "Cos'è la distrofia muscolare di Duchenne?"

    def test_translate_en_zh_CN(self, google_translator):
        translation = _translate(google_translator, "What is Duchenne Muscular Dystrophy?", source="en", target="zh-CN")
        assert translation == "什么是杜氏肌营养不良症？"

    def test_translate_en_zh_TW(self, google_translator):
        translation = _translate(google_translator, "What is Duchenne Muscular Dystrophy?", source="en", target="zh-TW")
        assert translation == "什麼是杜氏肌肉營養不良症？"


class TestOpusMTTranslator:
    def test_translate_fr_en(self, opusmt_translator):
        translation = _translate(
            opusmt_translator, "Qu'est-ce que la dystrophie musculaire de Duchenne?", source="fr", target="en"
        )
        assert translation == "What is Duchenne's Muscle Dystrophy?"

    def test_translate_it_en(self, opusmt_translator):
        translation = _translate(
            opusmt_translator, "Cos'è la distrofia muscolare di Duchenne?", source="it", target="en"
        )
        assert translation == "What is Duchenne's muscular dystrophy?"

    def test_translate_zh_CN_en(self, opusmt_translator):
        translation = _translate(opusmt_translator, "什么是杜氏肌营养不良症？", source="zh", target="en")
        assert translation == "What's Dow's malnourishment?"

    def test_translate_zh_TW_en(self, opusmt_translator):
        translation = _translate(opusmt_translator, "什麼是杜氏肌肉營養不良症？", source="zh", target="en")
        assert translation == "What's Dow's muscle nutrients?"

    def test_translate_en_fr(self, opusmt_translator):
        translation = _translate(opusmt_translator, "What is Duchenne Muscular Dystrophy?", source="en", target="fr")
        assert translation == "Qu'est-ce que la dystrophie musculaire de Duchenne?"

    def test_translate_en_it(self, opusmt_translator):
        translation = _translate(opusmt_translator, "What is Duchenne Muscular Dystrophy?", source="en", target="it")
        assert translation == "Che cos'è la distrofia muscolare di Duchenne?"

    def test_translate_en_zh_CN(self, opusmt_translator):
        translation = _translate(opusmt_translator, "What is Duchenne Muscular Dystrophy?", source="en", target="zh-CN")
        assert translation == "什么是杜尚尼亚肌肉萎缩症?"

    def test_translate_en_zh_TW(self, opusmt_translator):
        translation = _translate(opusmt_translator, "What is Duchenne Muscular Dystrophy?", source="en", target="zh-TW")
        assert translation == "什麼是杜尚尼亞肌肉萎縮癥?"
