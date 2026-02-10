import json
from argparse import ArgumentParser

import pandas as pd

from jndc.constants import Character, NovelFormat, Utterance


def main():
    parser = ArgumentParser()
    parser.add_argument("novel_path", type=str)
    parser.add_argument("character_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    with open(args.character_path, "r", encoding="utf-8") as f:
        chracter_list = json.load(f)
    character_name_dict = {chracter[Character.ID]: chracter[Character.NAME][0] for chracter in chracter_list}
    character_name_dict[-1] = "不明"
    novel_df = pd.read_csv(args.novel_path)
    dialogue_corups_list = []
    group_list = novel_df[NovelFormat.GROUP].tolist()
    for group in set(group_list):
        if group != -1:
            group_data = novel_df[novel_df[NovelFormat.GROUP] == group]
            dialogue_corups_list.append(
                {
                    "group": group,
                    "utterances": [
                        {
                            Utterance.UTTERANCE: sentence[1:-1],
                            Utterance.SPEKAERID: speaker,
                            Utterance.SPEKAERNAME: character_name_dict[speaker],
                        }
                        for sentence, speaker in zip(group_data[NovelFormat.SENTENCE], group_data[NovelFormat.SPEAKER])
                    ],
                }
            )
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(dialogue_corups_list, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
