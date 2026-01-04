import json
import os
from argparse import ArgumentParser

import pandas as pd

from jndc.character_name_list import NovelCharacterList
from jndc.config import MODEL_CONFIG
from jndc.constants import NovelFormat

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    parser = ArgumentParser()
    parser.add_argument("novel_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    novel_data = pd.read_csv(args.novel_path)
    character = NovelCharacterList(
        novel_text=novel_data[NovelFormat.SENTENCE].tolist(), ner_model_name=MODEL_CONFIG["ginza_path"]
    )
    character_list = character.create_charecter_list()
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(character_list, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
