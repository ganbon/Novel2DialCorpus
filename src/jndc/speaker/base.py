import re

from jndc.constants import Character


class BaseIdentifySpeaker:
    def __init__(self, character_name_list: list[dict]) -> None:
        self.character_name_list = character_name_list
        self.line_pattern = re.compile(r"「.+?」|『.+?』")

    def extract_sentence_character(self, sentence: str) -> list[int]:
        sentence_character = []
        for character in self.character_name_list:
            for name in character[Character.NAME]:
                if name in sentence:
                    sentence_character.append(character[Character.ID])
                    break
        return sentence_character

    def determine_line(self, sentence: str) -> bool:
        if self.line_pattern.match(sentence):
            return True
        else:
            return False

    def get_most_frequent_character(self, character_list: list[int]) -> int:
        candidate_count_list = [[candidate, character_list.count(candidate)] for candidate in set(character_list)]
        most_frequent_character = max(candidate_count_list, key=lambda x: x[1])
        if [candidate[1] for candidate in candidate_count_list].count(most_frequent_character[1]) == 1:
            return most_frequent_character[0]
        else:
            return -1
