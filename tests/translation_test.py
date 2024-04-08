import os

import pytest
from deep_translator import GoogleTranslator
from lingua import IsoCode639_1
from llama_index.llms.groq import Groq

from src.translation import _detect_language, _translate, get_language_detector
from translators.llm import LLMTranslator
from translators.opusmt import OpusMTTranslator
from translators.seamlessm4tv2 import SeamlessM4Tv2Translator


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


@pytest.fixture
def seamlessm4tv2_translator():
    return SeamlessM4Tv2Translator(source="fr", target="en")


@pytest.fixture
def mixtral8x7b_translator():
    os.environ["OPENAI_API_KEY"] = "None"
    llm = Groq(
        model="mixtral-8x7b-32768",
        api_key=os.environ["GROQ_API_KEY"],
    )
    languages = {
        "english": "en",
        "french": "fr",
        "german": "de",
        "italian": "it",
        "spanish": "es",
        "chinese (simplified)": "zh-CN",
        "chinese (traditional)": "zh-TW",
    }
    return LLMTranslator(llm, languages, source="fr", target="en")


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
        translation = _translate(opusmt_translator, "什么是杜氏肌营养不良症？", source="zh-CN", target="en")
        assert translation == "What's Dow's malnourishment?"

    def test_translate_zh_TW_en(self, opusmt_translator):
        translation = _translate(opusmt_translator, "什麼是杜氏肌肉營養不良症？", source="zh-TW", target="en")
        assert translation == "What's Dow's muscle nutrients?"

    def test_translate_en_fr(self, opusmt_translator):
        translation = _translate(opusmt_translator, "What is Duchenne Muscular Dystrophy?", source="en", target="fr")
        assert translation == "Qu'est-ce que la dystrophie musculaire de Duchenne?"

    def test_translate_en_fr_long(self, opusmt_translator):
        translation = _translate(
            opusmt_translator,
            "Summary of GRACILE syndrome: GRACILE syndrome is an inherited metabolic disease. GRACILE stands for growth retardation, aminoaciduria, cholestasis, iron overload, lactacidosis, and early death. Infants are very small at birth and quickly develop complications. During the first days of life, infants will develop a buildup of lactic acid in the bloodstream (lactic acidosis) and amino acids in the urine (aminoaciduria). They will also have problems with the flow of bile from the liver (cholestasis) and too much iron in their blood. Affected individuals aren't typically born with unique physical features. GRACILE syndrome is caused by a genetic change in the BCS1L gene, and it is inherited in an autosomal recessive pattern. The BCS1L gene provides instructions needed by the mitochondria in cells to help produce energy.",
            source="en",
            target="fr",
        )
        assert (
            translation
            == "Résumé du syndrome de GRACILE : Le syndrome de GRACILE est une maladie métabolique héréditaire. GRACILE représente un retard de croissance, une aminoacidité, une cholestase, une surcharge en fer, une acidocétose et une mort précoce. Les nourrissons sont très petits à la naissance et développent rapidement des complications. Au cours des premiers jours de la vie, les nourrissons développeront une accumulation d'acide lactique dans le flux sanguin (acidose lactique) et d'acides aminés dans l'urine (aminoacidurie). Ils auront également des problèmes avec le flux de bile du foie (cholestase) et trop de fer dans leur sang. Les personnes touchées ne sont généralement pas nées avec des caractéristiques physiques uniques. Le syndrome GRACILE est causé par un changement génétique du gène BCS1L, et il est hérité d'un motif récessif autosomal. Le gène BCS1L fournit les instructions nécessaires aux mitochondries dans les cellules pour aider à produire de l'énergie."
        )

    def test_translation_en_fr_overflow(self, opusmt_translator):
        translation = _translate(
            opusmt_translator,
            "Summary of Hirschsprung disease-deafness-polydactyly syndrome: Hirschsprung disease-deafness-polydactyly syndrome is an extremely rare malformative association, described in only two siblings to date, characterized by Hirschsprung disease (defined by the presence of an aganglionic segment of variable extent in the terminal part of the colon that leads to symptoms of intestinal obstruction, including constipation and abdominal distension), polydactyly of hands and/or feet, unilateral renal agenesis, hypertelorism and congenital deafness. There have been no further descriptions in the literature since 1988.",
            source="en",
            target="fr",
        )
        assert (
            translation
            == "Résumé du syndrome de la maladie de Hirschsprung : Le syndrome de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie est une association malformative extrêmement rare, décrite dans seulement deux frères et sœurs à ce jour, caractérisée par la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie de la maladie Il n'y a pas d'autres descriptions dans la littérature depuis 1988."
        )

    def test_translate_en_it(self, opusmt_translator):
        translation = _translate(opusmt_translator, "What is Duchenne Muscular Dystrophy?", source="en", target="it")
        assert translation == "Che cos'è la distrofia muscolare di Duchenne?"

    def test_translate_en_zh_CN(self, opusmt_translator):
        translation = _translate(opusmt_translator, "What is Duchenne Muscular Dystrophy?", source="en", target="zh-CN")
        assert translation == "什么是杜尚尼亚肌肉萎缩症?"

    def test_translate_en_zh_TW(self, opusmt_translator):
        translation = _translate(opusmt_translator, "What is Duchenne Muscular Dystrophy?", source="en", target="zh-TW")
        assert translation == "什麼是杜尚尼亞肌肉萎縮癥?"


class TestSeamlessM4Tv2Translator:
    def test_translate_fr_en(self, seamlessm4tv2_translator):
        translation = _translate(
            seamlessm4tv2_translator, "Qu'est-ce que la dystrophie musculaire de Duchenne?", source="fr", target="en"
        )
        assert translation == "What is Duchenne muscular dystrophy?"

    def test_translate_it_en(self, seamlessm4tv2_translator):
        translation = _translate(
            seamlessm4tv2_translator, "Cos'è la distrofia muscolare di Duchenne?", source="it", target="en"
        )
        assert translation == "What is Duchenne muscular dystrophy?"

    def test_translate_zh_CN_en(self, seamlessm4tv2_translator):
        translation = _translate(seamlessm4tv2_translator, "什么是杜氏肌营养不良症？", source="zh-CN", target="en")
        assert translation == "What is Duchenne muscular malnutrition?"

    def test_translate_zh_TW_en(self, seamlessm4tv2_translator):
        translation = _translate(seamlessm4tv2_translator, "什麼是杜氏肌肉營養不良症？", source="zh-TW", target="en")
        assert translation == "What is Duchenne muscular malnutrition?"

    def test_translate_en_fr(self, seamlessm4tv2_translator):
        translation = _translate(
            seamlessm4tv2_translator, "What is Duchenne Muscular Dystrophy?", source="en", target="fr"
        )
        assert translation == "Qu'est ce que la dystrophie musculaire de Duchenne?"

    def test_translate_en_it(self, seamlessm4tv2_translator):
        translation = _translate(
            seamlessm4tv2_translator, "What is Duchenne Muscular Dystrophy?", source="en", target="it"
        )
        assert translation == "Che cos'è la distrofia muscolare di Duchenne?"

    def test_translate_en_zh_CN(self, seamlessm4tv2_translator):
        translation = _translate(
            seamlessm4tv2_translator, "What is Duchenne Muscular Dystrophy?", source="en", target="zh-CN"
        )
        assert translation == "杜<unk>肌肉萎缩是什么?"

    def test_translate_en_zh_TW(self, seamlessm4tv2_translator):
        translation = _translate(
            seamlessm4tv2_translator, "What is Duchenne Muscular Dystrophy?", source="en", target="zh-TW"
        )
        assert translation == "杜<unk>肌肉萎缩是什么?"


class TestMixtral8x7bTranslator:
    def test_translate_fr_en(self, mixtral8x7b_translator):
        translation = _translate(
            mixtral8x7b_translator, "Qu'est-ce que la dystrophie musculaire de Duchenne?", source="fr", target="en"
        )
        assert translation == "What is Duchenne muscular dystrophy?"

    def test_translate_it_en(self, mixtral8x7b_translator):
        translation = _translate(
            mixtral8x7b_translator, "Cos'è la distrofia muscolare di Duchenne?", source="it", target="en"
        )
        assert translation == "What is Duchenne muscular dystrophy?"

    def test_translate_zh_CN_en(self, mixtral8x7b_translator):
        translation = _translate(mixtral8x7b_translator, "什么是杜氏肌营养不良症？", source="zh-CN", target="en")
        assert translation == "What is Duchenne muscular dystrophy (DMD)?"

    # def test_translate_zh_TW_en(self, mixtral8x7b_translator):
    #     translation = _translate(mixtral8x7b_translator, "什麼是杜氏肌肉營養不良症？", source="zh-TW", target="en")
    #     assert translation == "What is Duchenne muscular dystrophy (DMD)?"

    def test_translate_en_fr(self, mixtral8x7b_translator):
        translation = _translate(
            mixtral8x7b_translator, "What is Duchenne Muscular Dystrophy?", source="en", target="fr"
        )
        assert translation == "Qu'est-ce que la Dystrophie Musculaire de Duchenne ?"

    def test_translate_en_it(self, mixtral8x7b_translator):
        translation = _translate(
            mixtral8x7b_translator, "What is Duchenne Muscular Dystrophy?", source="en", target="it"
        )
        assert translation == "Che cos'è la distrofia muscolare di Duchenne?"

    def test_translate_en_zh_CN(self, mixtral8x7b_translator):
        translation = _translate(
            mixtral8x7b_translator, "What is Duchenne Muscular Dystrophy?", source="en", target="zh-CN"
        )
        assert translation == "杜<unk>肌肉萎缩是什么?"

    def test_translate_en_zh_TW(self, mixtral8x7b_translator):
        translation = _translate(
            mixtral8x7b_translator, "What is Duchenne Muscular Dystrophy?", source="en", target="zh-TW"
        )
        assert translation == "杜<unk>肌肉萎缩是什么?"
