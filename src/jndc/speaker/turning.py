import pandas as pd

from jndc.constants import NovelFormat, SpeakerCandidate
from jndc.speaker.base import BaseIdentifySpeaker


class TurningIdentifySpeaker(BaseIdentifySpeaker):
    def __init__(self, novel_data: pd.DataFrame, character_name_list: list[int], tone_llm_candidate: list[int]) -> None:
        super().__init__(character_name_list)
        self.novel_data = novel_data.copy()
        self.sentence_list = novel_data[NovelFormat.SENTENCE].tolist()
        self.group_list = novel_data[NovelFormat.GROUP].tolist()
        self.tone_llm_candidate = tone_llm_candidate

    def identify_turning_method(self) -> list[int]:
        previous_id = -1
        group_list, odd_list, even_list = [], [], []
        turning_candidate_list = [-1 for _ in range(len(self.novel_data))]
        for i, data in enumerate(zip(self.group_list, self.tone_llm_candidate)):
            group, candidate = data
            if previous_id + 1 == i and group != -1:
                group_list.append([i, candidate])
            elif len(group_list) > 2:
                for p, data in enumerate(group_list):
                    p_id, predict = data
                    if p % 2 and predict != -1:
                        odd_list.append(predict)
                    elif p % 2 == 0 and predict != -1:
                        even_list.append(predict)
                if odd_list == []:
                    odd_max = -1
                else:
                    odd_max = self.get_most_frequent_character(odd_list)
                if even_list == []:
                    even_max = -1
                else:
                    even_max = self.get_most_frequent_character(even_list)
                if odd_max == even_max:
                    if odd_list.count(odd_max) > even_list.count(even_list):
                        even_max = -1
                    elif odd_list.count(odd_max) < even_list.count(even_list):
                        odd_max = -1
                    else:
                        even_max, odd_max = -1, -1
                for p, data in enumerate(group_list):
                    p_id, predict = data
                    if p % 2 and odd_max != -1:
                        turning_candidate_list[p_id] = odd_max
                    elif p % 2 == 0 and even_max != -1:
                        turning_candidate_list[p_id] = even_max
                group_list, odd_list, even_list = [], [], []
            else:
                group_list, odd_list, even_list = [], [], []
            previous_id = i
        return turning_candidate_list

    def identify_turning_between_group(self) -> None:
        for group in range(max(self.group_list)):
            speaker_list = self.novel_data[SpeakerCandidate.SPEAKER][
                self.novel_data[NovelFormat.GROUP] == group
            ].tolist()
            group_index = self.novel_data[self.novel_data[NovelFormat.GROUP] == group].index
            if len(speaker_list) > 4:
                for i, speaker in enumerate(speaker_list):
                    if speaker == -1 and i >= 2 and i < len(speaker_list) - 2:
                        if (
                            speaker_list[i - 2] == speaker_list[i + 2]
                            and speaker_list[i - 1] == speaker_list[i + 1]
                            and speaker_list[i + 1] != speaker_list[i - 2]
                        ):
                            speaker_list[i] = speaker_list[i + 2]
            self.novel_data.loc[group_index, SpeakerCandidate.SPEAKER] = speaker_list

    def identify_speaker(self) -> tuple[list | None]:
        turning_candidate_list = self.identify_turning_method()
        speaker_list = [
            turn_candidate if speaker == -1 else speaker
            for turn_candidate, speaker in zip(turning_candidate_list, self.tone_llm_candidate)
        ]
        self.novel_data[SpeakerCandidate.SPEAKER] = speaker_list
        self.identify_turning_between_group()
        return self.novel_data
