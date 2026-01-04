from argparse import ArgumentParser

import pandas as pd

from jndc.concatenate_lines import LineConcatenate
from jndc.config import MODEL_CONFIG
from jndc.constants import NovelFormat


def main():
    parser = ArgumentParser()
    parser.add_argument("novel_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    novel_df = pd.read_csv(args.novel_path)
    concatenate_tool = LineConcatenate(novel_data=novel_df, model_path=MODEL_CONFIG["concat_model_path"])
    dialogue_group_list = concatenate_tool.concatenate_lines()
    novel_df[NovelFormat.GROUP] = dialogue_group_list
    novel_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
