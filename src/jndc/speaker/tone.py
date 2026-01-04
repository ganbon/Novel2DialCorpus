import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from jndc.constants import NovelFormat
from jndc.modules.morphological_analysis import MorphologicalAnalyzer
from jndc.speaker.base import BaseIdentifySpeaker
from jndc.speaker.utils import determine_line_format


class ToneIndetifySpeaker(BaseIdentifySpeaker):
    def __init__(
        self,
        novel_data: pd.DataFrame,
        character_name_list: list[dict],
        tone_model_path: str,
        pseudo_character_label_list: list[int],
        main_charactor_define_num: int = 20,
    ) -> None:
        super().__init__(character_name_list)
        self.novel_data = novel_data.copy()
        self.morphological = MorphologicalAnalyzer()
        self.sentence_list = novel_data[NovelFormat.SENTENCE].tolist()
        self.group_list = novel_data[NovelFormat.GROUP].tolist()
        self.character_name_list = character_name_list
        self.novel_data["pseudo_label"] = pseudo_character_label_list
        # self.pseudo_character_label_list = pseudo_character_label_list
        self.main_charactor_define_num = main_charactor_define_num
        self.tone_model = SentenceTransformer(tone_model_path)

    def filter_candidate_by_dialogue_group(self) -> list[list[int]]:
        group_num = max(self.group_list)
        group_candidate_list = []
        for group in set(self.group_list):
            group_candidates = []
            if group == -1:
                continue
            if group == 0:
                group_start_id = 0
            else:
                group_start_id = self.novel_data[NovelFormat.ID][
                    self.novel_data[NovelFormat.GROUP] == group - 1
                ].tolist()[-1]
            if group == group_num:
                group_end_id = self.novel_data["id"].tolist()[-1]
            else:
                group_end_id = self.novel_data[self.novel_data[NovelFormat.GROUP] == group + 1]["id"].tolist()[0]
            sentence_in_group_range_list = self.novel_data[NovelFormat.SENTENCE][
                (self.novel_data["id"] > group_start_id) & (self.novel_data["id"] < group_end_id)
            ]
            for sentence in sentence_in_group_range_list:
                group_candidates += self.extract_sentence_character(sentence)
            group_candidate_list.append(group_candidates)
        return group_candidate_list

    def encode_line(
        self, line_list: list[str], pseudo_character_label_list: list[int]
    ) -> tuple[pd.DataFrame, list[int]]:
        pca = PCA(n_components=100)
        main_character_list, label_list = [], []
        main_character_list = [
            p_id
            for p_id in set(pseudo_character_label_list)
            if pseudo_character_label_list.count(p_id) > self.main_charactor_define_num and p_id >= 0
        ]
        for i, sentence in enumerate(line_list):
            if i == 0:
                vector_data = (
                    self.tone_model.encode(sentence[1:-1], convert_to_tensor=True)
                    .reshape(-1, 768)
                    .to("cpu")
                    .detach()
                    .numpy()
                    .copy()
                )
            else:
                vector_data = np.concatenate(
                    [
                        vector_data,
                        self.tone_model.encode(sentence[1:-1], convert_to_tensor=True)
                        .reshape(-1, 768)
                        .to("cpu")
                        .detach()
                        .numpy()
                        .copy(),
                    ]
                )
            label_list.append(pseudo_character_label_list[i])
        pca.fit(np.array(vector_data))
        pca_vector = pca.transform(np.array(vector_data))
        vector_df = pd.DataFrame({"vector": [pca_vec for pca_vec in pca_vector], "character": label_list})
        return vector_df, main_character_list

    def create_representative_vector(self, vector_df: pd.DataFrame, main_character_list: list[int]) -> list:
        representative_vector_list = []
        for character in main_character_list:
            chracter_df = vector_df[vector_df["character"] == character]
            representative_vectors = chracter_df["vector"].tolist()
            representative_vector_list.append(np.mean(representative_vectors, axis=0))
        return representative_vector_list

    def defilne_character_threshold(
        self, vector_df: pd.DataFrame, main_character_list: list[int]
    ) -> tuple[list[float], list[float]]:
        soft_character_threshold_dict, hard_character_threshold_dict = {}, {}
        for character in main_character_list:
            person_data = vector_df[vector_df["character"] == character]
            vector_list = person_data["vector"].tolist()
            base_vector = np.mean(vector_list, axis=0)
            distance = cosine_similarity([np.array(base_vector)], vector_list)[0]
            q1, q3 = np.percentile(distance, [25, 75])
            soft_character_threshold_dict[character] = q1
            hard_character_threshold_dict[character] = q3
        return soft_character_threshold_dict, hard_character_threshold_dict

    def get_vector_similarity(
        self, vector_df: pd.DataFrame, representative_vector_list, main_character_list: list[int]
    ):
        target_vectors = np.array(vector_df["vector"].tolist())
        vector_similarity = cosine_similarity(target_vectors, np.array(representative_vector_list))
        return [{main_character_list[i]: cos_sim for i, cos_sim in enumerate(similar)} for similar in vector_similarity]

    def identify_speaker(self) -> tuple[list[int], list[int]]:
        line_df = self.novel_data[self.novel_data[NovelFormat.GROUP] >= 0]
        line_df = line_df[
            [determine_line_format(self.morphological, sentence[1:-1]) for sentence in line_df[NovelFormat.SENTENCE]]
        ]
        dialogue_group_candidate = self.filter_candidate_by_dialogue_group()
        tone_candidate_list, tone_similar_list = (
            [[] for _ in range(len(self.novel_data))],
            [[] for _ in range(len(self.novel_data))],
        )
        vector_df, main_character_list = self.encode_line(
            line_df[NovelFormat.SENTENCE].tolist(), line_df["pseudo_label"].tolist()
        )
        if main_character_list == []:
            return [], []
        representative_vector_list = self.create_representative_vector(vector_df, main_character_list)
        vector_candidate_list = self.get_vector_similarity(vector_df, representative_vector_list, main_character_list)
        soft_threshold, hard_threshold = self.defilne_character_threshold(vector_df, main_character_list)
        index = 0
        for i, sentence in enumerate(self.sentence_list):
            if determine_line_format(self.morphological, sentence[1:-1]) and self.group_list[i] >= 0:
                soft_tone_candidate = [
                    charactor
                    for charactor, similarity in vector_candidate_list[index].items()
                    if similarity > soft_threshold[charactor]
                    and charactor in dialogue_group_candidate[self.group_list[i]]
                ]
                hard_tone_candidate = [
                    charactor
                    for charactor, similarity in vector_candidate_list[index].items()
                    if similarity > hard_threshold[charactor]
                    and charactor in dialogue_group_candidate[self.group_list[i]]
                ]
                tone_candidate_list[i] = soft_tone_candidate
                tone_similar_list[i] = hard_tone_candidate
                index += 1
        return tone_candidate_list, tone_similar_list
