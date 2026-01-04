import json
from argparse import ArgumentParser

import pandas as pd

from jndc.config import MODEL_CONFIG
from jndc.constants import NovelFormat, SpeakerCandidate
from jndc.speaker.llm import LLMIdentifySpeaker
from jndc.speaker.rule import RuleIdentifySpeaker
from jndc.speaker.tone import ToneIndetifySpeaker
from jndc.speaker.turning import TurningIdentifySpeaker
from jndc.speaker.utils import integrate_rule_and_llm, integrate_tone_and_llm


def vaildate_use_model(use_llm, use_ginza):
    if use_llm and use_ginza:
        raise ValueError("cannot use both llama and ginza")


def main():
    parser = ArgumentParser()
    parser.add_argument("novel_path", type=str)
    parser.add_argument("character_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--use_ginza", action="store_true")
    args = parser.parse_args()
    vaildate_use_model(args.use_llm, args.use_ginza)
    with open(args.character_path, "r", encoding="utf-8") as f:
        chracter_list = json.load(f)
    novel_df = pd.read_csv(args.novel_path)
    if args.use_ginza:
        rule_method = RuleIdentifySpeaker(
            novel_data=novel_df, character_name_list=chracter_list, ginza_path=MODEL_CONFIG["ginza_path"]
        )
        rule_spaker_candidate_list = rule_method.idnetify_speaker()
        novel_df[SpeakerCandidate.RULE_CAND] = rule_spaker_candidate_list
        novel_df.to_csv(args.output_path, index=False)
    elif args.use_llm:
        llm_method = LLMIdentifySpeaker(
            novel_data=novel_df, character_name_list=chracter_list, llm_model_path=MODEL_CONFIG["llm_model_path"]
        )
        llm_speaker_candidate_list = llm_method.identify_speaker()
        novel_df[SpeakerCandidate.LLM_CAND] = llm_speaker_candidate_list
        novel_df.to_csv(args.output_path, index=False)
    if (
        SpeakerCandidate.RULE_CAND in novel_df.columns
        and SpeakerCandidate.LLM_CAND in novel_df.columns
        and not args.use_ginza
    ):
        rule_spaker_candidate_list = novel_df[SpeakerCandidate.RULE_CAND].tolist()
        llm_speaker_candidate_list = novel_df[SpeakerCandidate.LLM_CAND].tolist()
        rule_and_llm_candidate_list = integrate_rule_and_llm(rule_spaker_candidate_list, llm_speaker_candidate_list)
        tone_method = ToneIndetifySpeaker(
            novel_data=novel_df,
            character_name_list=chracter_list,
            tone_model_path=MODEL_CONFIG["tone_model_path"],
            pseudo_character_label_list=rule_and_llm_candidate_list,
        )
        tone_spaker_candidate_list, tone_spaker_info_list = tone_method.identify_speaker()
        novel_df[SpeakerCandidate.TONE_CAND] = tone_spaker_candidate_list
        novel_df[SpeakerCandidate.TONE_INFO] = tone_spaker_info_list
        llm_tone_candidate_list = integrate_tone_and_llm(
            sentence_list=novel_df[NovelFormat.SENTENCE].tolist(),
            rule_and_llm_candidate_list=rule_and_llm_candidate_list,
            llm_candidate_list=llm_speaker_candidate_list,
            tone_candidate_list=tone_spaker_candidate_list,
            tone_similar_list=tone_spaker_info_list,
            morphological=tone_method.morphological,
        )
        turning_method = TurningIdentifySpeaker(
            novel_data=novel_df, character_name_list=chracter_list, tone_llm_candidate=llm_tone_candidate_list
        )
        novel_df = turning_method.identify_speaker()
        novel_df[
            [
                NovelFormat.ID,
                NovelFormat.SECTION,
                NovelFormat.GROUP,
                NovelFormat.SPEAKER,
                NovelFormat.SENTENCE,
                SpeakerCandidate.RULE_CAND,
                SpeakerCandidate.LLM_CAND,
                SpeakerCandidate.TONE_CAND,
                SpeakerCandidate.TONE_INFO,
            ]
        ].to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
