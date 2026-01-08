import itertools

import ginza
import pandas as pd
import spacy

from jndc.constants import NovelFormat
from jndc.modules.morphological_analysis import MorphologicalAnalyzer
from jndc.speaker.base import BaseIdentifySpeaker


class RuleIdentifySpeaker(BaseIdentifySpeaker):
    def __init__(
        self,
        novel_data: pd.DataFrame,
        character_name_list: list[dict],
        ginza_path: str,
    ) -> None:
        super().__init__(character_name_list)
        self.novel_data = novel_data.copy()
        self.morphological = MorphologicalAnalyzer()
        self.sentence_list = novel_data[NovelFormat.SENTENCE].tolist()
        self.group_list = novel_data[NovelFormat.GROUP].tolist()
        self.nlp = spacy.load(ginza_path)
        spacy.prefer_gpu()

    def extract_nsubj(self, sentence: str) -> list[str]:
        nmod_list, subj_list = [], []
        extract_word = []
        doc = self.nlp(sentence)
        for span in ginza.bunsetu_phrase_spans(doc):
            for token in span.lefts:
                if token.dep_ == "nsubj":
                    subj_list.append([str(span), str(ginza.bunsetu_span(token))])
                elif token.dep_ == "nmod":
                    nmod_list.append([str(span), str(ginza.bunsetu_span(token))])
        for subj in subj_list:
            for nmod in nmod_list:
                if nmod[0] == subj[0]:
                    extract_word.append(nmod[1])
            extract_word.append(subj[1])
        return extract_word

    def remove_inline_character(self) -> list[list[int]]:
        label_list = [[] for _ in self.sentence_list]
        for i, sentence in enumerate(self.sentence_list):
            if self.group_list[i] != -1:
                sentence_character = set(self.extract_sentence_character(sentence))
                label_list.append(sentence_character)
        return label_list

    def extract_calling_spans(self, sentence: str) -> list[list[int]]:
        part_sentence = ""
        extract_name_list = []
        token_list = self.morphological.segment_text_into_morphemes(sentence)
        feature_dict = self.morphological.get_morpheme2feature_dict(sentence)
        for token in token_list:
            if feature_dict[token].pos == "名詞" or feature_dict[token].pos == "接尾辞":
                part_sentence += token
            elif feature_dict[token].pos == "補助記号" and part_sentence != "":
                extract_name_list += self.extract_sentence_character(part_sentence)
                part_sentence = ""
            else:
                part_sentence = ""
        return extract_name_list

    def narretion_pattern(self) -> list[list[int]]:
        label_list = [[] for _ in self.sentence_list]
        for i, sentence in enumerate(self.sentence_list):
            subject_list = []
            if i > 1:
                if (
                    not self.determine_line(self.sentence_list[i - 2])
                    and not self.determine_line(self.sentence_list[i - 1])
                    and self.group_list[i] != -1
                ):
                    subject_list += self.extract_nsubj(self.sentence_list[i - 1])
            if i < len(self.sentence_list) - 2:
                if (
                    not self.determine_line(self.sentence_list[i + 1])
                    and not self.determine_line(self.sentence_list[i + 2])
                    and self.group_list[i] != -1
                ):
                    subject_list += self.extract_nsubj(self.sentence_list[i + 1])
            identify_character_id = list(
                itertools.chain.from_iterable([self.extract_sentence_character(subject) for subject in subject_list])
            )
            label_list[i] = identify_character_id
        return label_list

    def line_content_pattern(self) -> list[list[int]]:
        label_list = [[] for _ in self.sentence_list]
        previous_group_index, previous_line, previous_id = -100, None, 0
        for i, sentence in enumerate(self.sentence_list):
            if previous_group_index == self.group_list[i]:
                label_list[i] += self.extract_calling_spans(previous_line[1:-1])
            if previous_group_index == self.group_list[i]:
                label_list[previous_id] += self.extract_calling_spans(sentence[1:-1])
            if self.group_list[i] != -1:
                previous_group_index = self.group_list[i]
                previous_id = i
                previous_line = sentence
        return label_list

    def idnetify_speaker(self) -> list[int]:
        rule_method_candidate_list = [-1 for _ in range(len(self.novel_data))]
        speaker_candidate_by_narration_pattern_list = self.narretion_pattern()
        speaker_candidate_by_line_pattern_list = self.line_content_pattern()
        remove_charcter_list = self.remove_inline_character()
        for i, candidates in enumerate(
            zip(
                speaker_candidate_by_narration_pattern_list,
                speaker_candidate_by_line_pattern_list,
                remove_charcter_list,
            )
        ):
            narration_pattern, line_pattern, remove_character = candidates
            candidate_list = [
                candidate for candidate in narration_pattern + line_pattern if candidate not in remove_character
            ]
            if candidate_list != []:
                rule_method_candidate_list[i] = self.get_most_frequent_character(candidate_list)
        return rule_method_candidate_list
