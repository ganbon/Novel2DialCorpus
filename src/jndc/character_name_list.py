import itertools

import spacy

from jndc.constants import Character


class NovelCharacterList:
    def __init__(self, novel_text: str | list[str], ner_model_name: str = "ginza_path", filter_count: int = 10) -> None:
        if isinstance(novel_text, str):
            self.novel_text = novel_text.split("\n")
        else:
            self.novel_text = novel_text
        spacy.prefer_gpu()
        self.nlp = spacy.load(ner_model_name)
        self.filter_count = filter_count

    def extract_character_name(self, sentence) -> list[str]:
        doc = self.nlp(sentence)
        return [ent.text for ent in doc.ents if ent.label_ == "Person"]

    def count_character_name(self, character_name_list: list[str]) -> dict[str, int]:
        name_count_dict = {
            character_name: character_name_list.count(character_name) for character_name in set(character_name_list)
        }
        filter_name_dict = {
            character_name: count for character_name, count in name_count_dict.items() if count >= self.filter_count
        }
        return filter_name_dict

    def remove_duplicated_group(self, character_name_group_dict: dict) -> dict[str, list[str]]:
        remove_duplicated_name_group_dict = character_name_group_dict.copy()
        remove_key_list = []
        for target_name, target_name_group in character_name_group_dict.items():
            if target_name in remove_key_list:
                continue
            loop_start = False
            for ref_name, ref_name_group in character_name_group_dict.items():
                if target_name == ref_name:
                    loop_start = True
                    continue
                if loop_start and set(ref_name_group) - set(target_name_group) == set():
                    remove_duplicated_name_group_dict.pop(ref_name)
                    remove_key_list.append(ref_name)
        return remove_duplicated_name_group_dict

    def get_name_group(self, count_character_name_dict: dict) -> dict[str, list[str]]:
        character_name_list = sorted(list(count_character_name_dict.keys()), key=lambda x: len(x), reverse=True)
        character_name_group_dict = {}
        for target_name in character_name_list:
            character_name_group_dict[target_name] = [
                candidate_name for candidate_name in character_name_list if candidate_name in target_name
            ]
        return character_name_group_dict

    def reorganaize_name_group(
        self, character_name_group_dict: dict, count_character_name_dict: dict
    ) -> dict[str, list[str]]:
        contained_same_name_pair_list = []
        reorganaize_name_group_dict = character_name_group_dict.copy()
        for target_name, target_name_group in character_name_group_dict.items():
            loop_start = False
            for ref_name, ref_name_group in character_name_group_dict.items():
                if target_name == ref_name:
                    loop_start = True
                    continue
                if loop_start and set(target_name_group) & set(ref_name_group) != set():
                    contained_same_name_pair_list.append(
                        (target_name, ref_name, set(target_name_group) & set(ref_name_group))
                    )
        for name1, name2, same_set in contained_same_name_pair_list:
            name1_count = sum([count_character_name_dict[name] for name in character_name_group_dict[name1]])
            name2_count = sum([count_character_name_dict[name] for name in character_name_group_dict[name2]])
            if name1_count > name2_count:
                reorganaize_name_group_dict[name2] = list(set(reorganaize_name_group_dict[name2]) - same_set)
            else:
                reorganaize_name_group_dict[name1] = list(set(reorganaize_name_group_dict[name1]) - same_set)
        return reorganaize_name_group_dict

    def create_charecter_list(self) -> list[dict]:
        character_name_list = list(
            itertools.chain.from_iterable([self.extract_character_name(sentence) for sentence in self.novel_text])
        )
        count_character_name_dict = self.count_character_name(character_name_list)
        character_name_group_dict = self.get_name_group(count_character_name_dict)
        remove_duplicated_name_group_dict = self.remove_duplicated_group(character_name_group_dict)
        reorganaize_name_group_dict = self.reorganaize_name_group(
            remove_duplicated_name_group_dict, count_character_name_dict
        )
        character_name_group_list = [
            {
                Character.ID: i,
                Character.NAME: name_group,
                Character.COUNT: sum([count_character_name_dict[name] for name in name_group]),
            }
            for i, name_group in enumerate(reorganaize_name_group_dict.values())
        ]
        sorted_character_name_group_list = sorted(
            character_name_group_list, key=lambda x: x[Character.COUNT], reverse=True
        )
        for i in range(len(sorted_character_name_group_list)):
            sorted_character_name_group_list[i][Character.ID] = i
        return sorted_character_name_group_list
