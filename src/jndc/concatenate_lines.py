import re
from enum import IntEnum

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, BertJapaneseTokenizer

from jndc.constants import NovelFormat


class ExtractionOption(IntEnum):
    PUSH = 1
    SKIP = 2
    COMP = 3


class LineConcatenate:
    def __init__(
        self, novel_data: pd.DataFrame, model_path: str, between_length: int = 2, skip_between_length: int = 9
    ) -> None:
        self.novel_data = novel_data
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True).to(
            self.device
        )
        self.bert_tokenizer = BertJapaneseTokenizer.from_pretrained(model_path)
        self.between_length = between_length
        self.skip_between_length = skip_between_length
        self.line_pattern = re.compile(r"「.+?」|『.+?』")

    def determine_relationship(self, previous_sentence: str, target_sentence: str) -> int:
        encoding = self.bert_tokenizer(
            previous_sentence, target_sentence, max_length=512, padding="max_length", return_tensors="pt"
        )
        if len(encoding["input_ids"][0]) > 512:
            return 1
        output = self.bert_model(
            input_ids=encoding["input_ids"].to(self.device), token_type_ids=encoding["token_type_ids"].to(self.device)
        )
        return torch.argmax(output["logits"]).item()

    def delete_between_sentence(self, previous_sentence: list[str]) -> list[int]:
        previous_line_list = [sentence for sentence in previous_sentence if self.line_pattern.fullmatch(sentence)]
        between_sentence_list = [
            sentence for sentence in previous_sentence if not self.line_pattern.fullmatch(sentence)
        ]
        if len(between_sentence_list) > self.between_length * 2:
            return (
                previous_line_list
                + between_sentence_list[: self.between_length]
                + between_sentence_list[-self.between_length :]
            )
        else:
            return previous_sentence

    def confirm_extraction_target(self, target_sentence: str, surrounding_sentence_list: list[str]) -> int:
        between_sentence_list = [
            sentence for sentence in surrounding_sentence_list if not self.line_pattern.fullmatch(sentence)
        ]
        if surrounding_sentence_list == [] and not self.line_pattern.fullmatch(target_sentence):
            return ExtractionOption.SKIP
        elif between_sentence_list != [] and self.line_pattern.fullmatch(target_sentence):
            return ExtractionOption.COMP
        else:
            return ExtractionOption.PUSH

    def define_dialogue_group_index(self, sentnece, dialogue_group_index) -> int:
        if self.line_pattern.fullmatch(sentnece):
            return dialogue_group_index
        else:
            return -1

    def correct_dialogue_group_index(self, group_index_list: list[int]) -> list[int]:
        new_index_dict = {}
        index_count = 0
        for group_id in set(group_index_list):
            if group_index_list.count(group_id) > 1 and group_id != -1:
                new_index_dict[group_id] = index_count
                index_count += 1
            else:
                new_index_dict[group_id] = -1
        return [new_index_dict[index] for index in group_index_list]

    def concatenate_lines(self) -> list[int]:
        surrounding_sentence_list, group_index_list = [], []
        group_index = 0
        sections = self.novel_data[NovelFormat.SECTION].tolist()
        for i, sentence in tqdm(enumerate(self.novel_data[NovelFormat.SENTENCE]), total=len(self.novel_data)):
            if sections[i] != max(sections) and sections[i] != sections[i + 1]:
                group_index += 1
                if self.line_pattern.fullmatch(sentence):
                    surrounding_sentence_list = [sentence]
                else:
                    surrounding_sentence_list = []
            if self.confirm_extraction_target(sentence, surrounding_sentence_list) == ExtractionOption.PUSH:
                surrounding_sentence_list.append(sentence)
            elif self.confirm_extraction_target(sentence, surrounding_sentence_list) == ExtractionOption.COMP:
                between_sentence_list = [
                    sentence for sentence in surrounding_sentence_list if not self.line_pattern.fullmatch(sentence)
                ]
                if len(between_sentence_list) <= self.skip_between_length:
                    previous_sentence = "[SEP]".join(self.delete_between_sentence(surrounding_sentence_list))
                    prediction = self.determine_relationship(previous_sentence, sentence)
                    if prediction == 1:
                        group_index += 1
                else:
                    group_index += 1
                surrounding_sentence_list = [sentence]
            group_index_list.append(self.define_dialogue_group_index(sentence, group_index))
        group_index_list = self.correct_dialogue_group_index(group_index_list)
        return group_index_list
