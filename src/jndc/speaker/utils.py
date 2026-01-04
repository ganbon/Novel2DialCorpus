from jndc.modules.morphological_analysis import MorphologicalAnalyzer


def integrate_rule_and_llm(rule_candidate_list, llm_candidate_list) -> list[int]:
    rule_and_llm_candidate_list = []
    for rule_candidate, llm_candidate in zip(rule_candidate_list, llm_candidate_list):
        if rule_candidate != -1 and llm_candidate != -1 and rule_candidate == llm_candidate:
            rule_and_llm_candidate_list.append(rule_candidate)
        else:
            rule_and_llm_candidate_list.append(-1)
    return rule_and_llm_candidate_list


def determine_line_format(morphological: MorphologicalAnalyzer, sentence: str, length=4) -> bool:
    token_list = morphological.segment_text_into_morphemes(sentence)
    feature_dict = morphological.get_morpheme2feature_dict(sentence)
    feature_list = [feature_dict[token].pos for token in token_list]
    if ("助動詞" not in feature_list and "助詞" not in feature_list) or len(token_list) < length:
        return False
    else:
        return True


def integrate_tone_and_llm(
    sentence_list: list[str],
    rule_and_llm_candidate_list: list,
    llm_candidate_list: list,
    tone_candidate_list: list[list[int]],
    tone_similar_list: list[dict],
    morphological: MorphologicalAnalyzer,
    main_charactor_define_num: int = 20,
):
    main_person_list = [
        p_id
        for p_id in set(rule_and_llm_candidate_list)
        if rule_and_llm_candidate_list.count(p_id) > main_charactor_define_num and p_id >= 0
    ]
    integrate_tone_and_llm_candidate_list = rule_and_llm_candidate_list
    for i, data in enumerate(
        zip(
            sentence_list,
            rule_and_llm_candidate_list,
            llm_candidate_list,
            tone_candidate_list,
            tone_similar_list,
        )
    ):
        sentence, speaker, llm_candidate, tone_candidate, hard_tone_candidate = data
        if speaker == -1:
            if (
                llm_candidate in tone_candidate
                or llm_candidate not in main_person_list
                or not determine_line_format(morphological, sentence[1:-1])
            ):
                integrate_tone_and_llm_candidate_list[i] = llm_candidate
            elif len(hard_tone_candidate) == 1:
                integrate_tone_and_llm_candidate_list[i] = hard_tone_candidate[0]
    return integrate_tone_and_llm_candidate_list
