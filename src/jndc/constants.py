from enum import StrEnum


class NovelFormat(StrEnum):
    ID = "id"
    SECTION = "section"
    GROUP = "group"
    SENTENCE = "sentence"
    SPEAKER = "speaker"


class SpeakerCandidate(StrEnum):
    PATTERN_NA = "pattern_narration"
    PATTERN_LN = "pattern_line"
    LLM_CAND = "llm_candidate"
    RULE_CAND = "rule_candidate"
    RULE_LLM_CAND = "rule_llm_candidate"
    TONE_CAND = "tone_candidate"
    TONE_INFO = "tone_sim_info"
    TONE_LLM_CAND = "tone_llm_candidate"
    SPEAKER = "speaker"


class Character(StrEnum):
    ID = "id"
    NAME = "name"
    COUNT = "count"


class Utterance(StrEnum):
    UTTERANCE = "utterance"
    SPEKAERID = "speaker_id"
    SPEKAERNAME = "speaker_name"
