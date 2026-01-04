import itertools
import os
import re
from glob import glob
from unicodedata import normalize

import pandas as pd
import regex

from jndc.constants import NovelFormat


class PreProcess:
    def __init__(
        self, novel_path: str | None = None, novel_dir: str | None = None, skip_filename: list[str] | None = None
    ) -> None:
        self.novel_path = novel_path
        self.novel_dir = novel_dir
        self.skip_filename = skip_filename
        self.line_pattern = re.compile(r"「.+?」|『.+?』")

    def load_novel(self, file_path: str) -> list[str]:
        with open(file_path, mode="r", encoding="utf-8") as f:
            novel_text_list = [line.strip() for line in f]
        return novel_text_list

    def nomalize_sentence(self, sentence: str) -> str:
        remove_pattern = re.compile(r"[\t\u3000]")
        normalized_sentence = remove_pattern.sub("", normalize("NFKC", sentence))
        return normalized_sentence

    def skip_file_loading(self, file_path: str) -> bool:
        for name in self.skip_filename:
            if name in file_path:
                return True
        return False

    def determine_only_sinbols_sentence(self, sentence: str) -> bool:
        symbols_pattern = regex.compile(r"^[\p{P}\p{S}]+$")
        if symbols_pattern.match(sentence):
            return True
        else:
            return False

    def add_line_break(self, sentence: str) -> str:
        if self.line_pattern.fullmatch(sentence):
            return sentence
        elif self.line_pattern.search(sentence):
            line_search_data = self.line_pattern.finditer(sentence)
            for line_group in reversed(list(line_search_data)):
                span = line_group.span()
                sentence = sentence[: span[0]] + "\n" + sentence[span[0] : span[1]] + "\n" + sentence[span[1] :]
        else:
            sentence = sentence.replace("。", "。\n")
        return sentence

    def create_base_format(self) -> pd.DataFrame:
        base_format_list = []
        section = 0
        if self.novel_path and os.path.exists(self.novel_path):
            novel_text = self.load_novel(self.novel_path)
            novel_text = [self.add_line_break(sentence) for sentence in self.load_novel(self.novel_path)]
            split_novel_text = list(itertools.chain.from_iterable([sentence.split("\n") for sentence in novel_text]))
            base_format_list = [
                {
                    NovelFormat.ID: id,
                    NovelFormat.SECTION: section,
                    NovelFormat.SENTENCE: self.nomalize_sentence(sentence),
                }
                for id, sentence in enumerate(split_novel_text)
                if sentence and not self.determine_only_sinbols_sentence(sentence)
            ]
        if self.novel_dir and os.path.exists(self.novel_dir):
            file_path_list = sorted(
                [
                    (int(os.path.splitext(os.path.basename(novel_path))[0]), novel_path)
                    for novel_path in glob(self.novel_dir + "*.txt")
                ]
            )
            for _, novel_path in file_path_list:
                total_num = len(base_format_list)
                if self.skip_file_loading(novel_path):
                    continue
                novel_text = [self.add_line_break(sentence) for sentence in self.load_novel(novel_path) if sentence]
                novel_sentence_list = list(
                    itertools.chain.from_iterable([sentence.split("\n") for sentence in novel_text])
                )
                novel_sentence_list = [
                    sentence
                    for sentence in novel_sentence_list
                    if sentence and not self.determine_only_sinbols_sentence(sentence)
                ]
                base_format_list += [
                    {
                        NovelFormat.ID: id + total_num,
                        NovelFormat.SECTION: section,
                        NovelFormat.SENTENCE: self.nomalize_sentence(sentence),
                    }
                    for id, sentence in enumerate(novel_sentence_list)
                ]
                section += 1
        return pd.DataFrame(base_format_list)
