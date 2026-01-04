from argparse import ArgumentParser
from pathlib import Path

from jndc.preprocess import PreProcess


def main():
    parser = ArgumentParser()
    parser.add_argument("novel_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    novel_path = args.novel_path
    if Path(novel_path).is_file():
        process_tool = PreProcess(novel_path=novel_path, skip_filename=[])
    if Path(novel_path).is_dir():
        process_tool = PreProcess(novel_dir=novel_path, skip_filename=[])
    novel_df = process_tool.create_base_format()
    novel_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
