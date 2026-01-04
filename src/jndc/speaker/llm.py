import ast

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from jndc.constants import NovelFormat
from jndc.modules.morphological_analysis import MorphologicalAnalyzer
from jndc.speaker.base import BaseIdentifySpeaker


class LLMIdentifySpeaker(BaseIdentifySpeaker):
    def __init__(
        self,
        novel_data: pd.DataFrame,
        character_name_list: list[dict],
        llm_model_path: str,
        ground_length: int = 10,
    ) -> None:
        super().__init__(character_name_list)
        self.novel_data = novel_data.copy()
        self.morphological = MorphologicalAnalyzer()
        self.sentence_list = novel_data[NovelFormat.SENTENCE].tolist()
        self.group_list = novel_data[NovelFormat.GROUP].tolist()
        self.character_name_list = character_name_list
        self.ground_length = ground_length
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            torch_dtype="auto",
            device_map="auto",
        )

    def create_template(self, ground_text: str, target_line: str):
        system_prompt = "あなたは日本語に詳しいアシスタントです。"
        text = f"""
        [小説本文]の文脈から[対象の台詞]の発話者の人物名を推測しなさい。
        発話者の人物名が不明な場合は「不明」と答えなさい。
        [小説本文]：{ground_text}
        [対象の台詞]：{target_line}
        出力フォーマット:{{"speaker":"推測した発話者"}}
        """
        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
        return message

    def identify_speaker(self) -> list[int]:
        llm_candidate_list = [-1 for _ in range(len(self.novel_data))]
        sentence_list = self.sentence_list.copy()
        for i, sentence in tqdm(enumerate(sentence_list), total=len(sentence_list)):
            if self.group_list[i] != -1:
                if len(sentence_list[i:]) <= self.ground_length:
                    ground_text = "\n".join(sentence_list[i - self.ground_length :])
                elif len(sentence_list[:i]) <= self.ground_length:
                    ground_text = "\n".join(sentence_list[: i + self.ground_length])
                else:
                    ground_text = "\n".join(sentence_list[i - self.ground_length : i + self.ground_length])
                prompt = self.llm_tokenizer.apply_chat_template(
                    self.create_template(ground_text, sentence), tokenize=False, add_generation_prompt=True
                )
                token_ids = self.llm_tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
                output_ids = self.llm_model.generate(
                    token_ids.to(self.llm_model.device),
                    max_new_tokens=1200,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.llm_tokenizer.pad_token_id,
                )
                output = self.llm_tokenizer.decode(
                    output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True
                )
                try:
                    output_json_fomat = ast.literal_eval(output)
                    prediction_character_id = self.extract_sentence_character(output_json_fomat["speaker"])
                    if prediction_character_id != []:
                        llm_candidate_list[i] = prediction_character_id[0]
                        sentence_list[i] = output_json_fomat["speaker"] + ":" + sentence
                except ValueError:
                    continue
        return llm_candidate_list
