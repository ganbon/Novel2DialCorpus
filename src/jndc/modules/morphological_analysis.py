from dataclasses import dataclass
from enum import IntEnum

from MeCab import Tagger
from pykakasi import kakasi


class Transliterator:
    def __init__(self):
        self.kks = kakasi()

    def transliterate_text(self, text: str) -> str:
        result = self.kks.convert(text)
        return "".join(word["hira"] for word in result)


@dataclass
class FeatureDict:
    pos: str
    subpos: tuple[str, str, str]
    conjform: str
    lemma: str
    reading: str


class Index(IntEnum):
    POS = 0
    SUBPOS1 = 1
    SUBPOS2 = 2
    SUBPOS3 = 3
    CONJTYPE = 4
    CONJFORM = 5
    LEMMA = 6
    READING = 7
    PRONUNCIATION = 8


class MorphologicalAnalyzer:
    def __init__(self) -> None:
        self.transliterator = Transliterator()
        self.tagger = Tagger()

    def segment_text_into_morphemes(self, text: str) -> list[str]:
        node = self.tagger.parseToNode(text)

        morphemes = []
        while node:
            morpheme = node.surface
            if morpheme:
                morphemes.append(morpheme)
            node = node.next
        return morphemes

    def get_morpheme2feature_dict(self, text: str) -> dict[str, FeatureDict]:
        node = self.tagger.parseToNode(text)

        morpheme2feature_dict = {}
        while node:
            morpheme = node.surface
            if morpheme not in morpheme2feature_dict.keys():
                features = node.feature.split(",")
                morpheme2feature_dict[morpheme] = FeatureDict(
                    pos=features[Index.POS],
                    subpos=(
                        features[Index.SUBPOS1],
                        features[Index.SUBPOS2],
                        features[Index.SUBPOS3],
                    ),
                    conjform=features[Index.CONJFORM],
                    lemma=features[Index.LEMMA] if len(features) >= 7 else "*",
                    reading=self.transliterator.transliterate_text(morpheme),
                )
            node = node.next
        return morpheme2feature_dict
