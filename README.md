# Novel2DialCorpus 
Novel2DialCorpusは日本語の小説テキストから台詞群を会話として抽出し，発話者を付与した雑談対話コーパスを自動構築する手法です。

## Output File
`dialogue_corpus.json`，`character.json`，`novel_data.csv`の3つのファイルを出力します。

### 1. dialogue_corpus.json
小説から構築した対話コーパスを保存したjsonファイル。
1会話は以下のフォーマットで保存されます。
| key | type | content |
| :--- | :--- | :--- |
| group |int | 会話のグループID |
| utterances | list[dict] | 会話の内容 |
| utterance | str | 発話 |
| spekaer_id | int | 発話者ID |
| speaker_name | str | 発話者名 |
```json
 {
        "group": 0,
        "uttreances": [
            {
                "utterance": "あ、太郎!朝だよ!",
                "speaker_id": 3,
                "speaker_name": "花子"
            },
            {
                "utterance": "おはよう、花子。いつも通りだね",
                "speaker_id": 0,
                "speaker_name": "太郎"
            },
            {
                "utterance": "そっちこそ寝ぼけた顔してるよ!今日は新学期の朝礼があるんだから、シャキッとしないと",
                "speaker_id": 3,
                "speaker_name": "花子"
            },
            {
                "utterance": "分かったよ。朝礼か......",
                "speaker_id": 0,
                "speaker_name": "太郎"
            },
            {
                "utterance": "おい、太郎!寝坊したのか?",
                "speaker_id": 2,
                "speaker_name": "次郎"
            },
            {
                "utterance": "次郎か。朝から元気だな",
                "speaker_id": 0,
                "speaker_name": "太郎"
            },
            {
                "utterance": "当たり前だろ。新学年だぜ。誰が新しいクラスメートか見物じゃないか",
                "speaker_id": 2,
                "speaker_name": "次郎"
            },
            {
                "utterance": "そういう次郎はどうなの?何か楽しみあるの?",
                "speaker_id": 3,
                "speaker_name": "花子"
            },
            {
                "utterance": "別に......ただ、新しい環境って刺激があるじゃないか。それだけだよな",
                "speaker_id": 2,
                "speaker_name": "次郎"
            },
            {
                "utterance": "そんなことより、朝礼の時間だ。行くぞ",
                "speaker_id": 0,
                "speaker_name": "太郎"
            }
        ]
    }
```

### character.josn
小説に登場する人物名を抽出したリスト
| key | type | content |
| :--- | :--- | :--- |
| id |int | 人物ID |
| name | list[str] | 名前集合 |
| count | str | 抽出数 |
```json
[
    {
        "id": 0,
        "name": [
            "太郎"
        ],
        "count": 239
    },
    {
        "id": 1,
        "name": [
            "桜子"
        ],
        "count": 154
    },
    {
        "id": 2,
        "name": [
            "次郎"
        ],
        "count": 41
    },
    {
        "id": 3,
        "name": [
            "花子"
        ],
        "count": 38
    }
]
```

### novel_data.csv
台詞に対して会話グループID、各手法で特定した発話者IDが付与されたcsvファイル

| key | type | content |
| :--- | :--- | :--- |
| group |int | 会話のグループID |
| utterances | list[dict] | 会話の内容 |
| utterance | str | 発話 |
| spekaer_id | int | 発話者ID |
| speaker_name | str | 発話者名 |
```csv
id,section,group,speaker,sentence,rule_candidate,llm_candidate,tone_candidate,tone_sim_info
0,0,-1,-1,春日高校の校門前。,-1,-1,[],[]
1,0,-1,-1,満開の桜が風に揺られ、淡紅色の花びらが舞い落ちる。,-1,-1,[],[]
2,0,-1,-1,朝日が差し込む教室。,-1,-1,[],[]
3,0,-1,-1,太郎は大きく伸びをしながら席に座った。,-1,-1,[],[]
4,0,0,3,「あ、太郎!朝だよ!」,-1,3,[],[]
5,0,-1,-1,元気な声をかけてきたのは、幼馴染の花子だ。,-1,-1,[],[]
6,0,-1,-1,黒髪をツインテールにしたその少女は、太郎の机に両手をついて身を乗り出している。,-1,-1,[],[]
7,0,0,0,「おはよう、花子。いつも通りだね」,0,0,[0],[0]
8,0,-1,-1,太郎は花子に笑いかけた。,-1,-1,[],[]
9,0,-1,-1,十年近く一緒にいるからか、この光景はもう日常そのものだ。,-1,-1,[],[]
10,0,0,3,「そっちこそ寝ぼけた顔してるよ!今日は新学期の朝礼があるんだから、シャキッとしないと」,3,3,[0],[]
```

## Data
Claude Haiku 3.5で生成した疑似小説をサンプルデータとして`./data/sample_novel/`に配置しています。

## Model
以下の3つのモデルを使用しています。
- 発話応答関係モデル：[ganbon/novel-bert-base-relationship](https://huggingface.co/ganbon/novel-bert-base-relationship)
- 口調ベクトルモデル：[ganbon/novel-sentence-bert-base-tone-embedding](https://huggingface.co/ganbon/novel-sentence-bert-base-tone-embedding)
- LLM：[tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.2](https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.2)

## Experimental Setup
### Device
- GPU:NVIDIA GeForce RTX 4090 16GB
### Python Enviroments
pythonの仮想環境ツールとして[uv](https://docs.astral.sh/uv/)が必要です
- uvのインストール
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
- 仮想環境の構築
```shell
$ ./scripts/create_venvs.sh
```

## Usage
### 1. Preprocess
```shell
$ source ./venv-w-ginza/bin/activate 
$ python ./scripts/create_corpus_format.py ./data/sample_novel/　./data/result/novel_data.csv
$ deactivate
```
### 2. Create Character List
小説内から人物名を抽出し，人物リストを作成します。
```shell
$ source ./venv-w-ginza/bin/activate 
$ python  ./scripts/create_character_name_list.py ./data/sample_novel/novel_data.csv　./data/result/character.json 
$ deactivate
```

> [!NOTE] 
> 人物リストは構築後，人手で確認，整理することをお勧めします。

### 3. Create Dialogue Group
各台詞で連続している台詞群を1会話として、会話グループを作成します。
2文以上連続している台詞群を会話とみなします。
```shell
$ source ./venv-w-ginza/bin/activate 
$ python ./scripts/assign_dialogue_group.py ./data/result/novel_data.csv　./data/result/novel_data.csv 
$ deactivate
```
### 4. Idetify Speaker
会話グループに属する台詞に対して発話者を特定します。
3つの手法を用いて発話者を特定します。
#### Rule Method
```shell
$ source ./venv-w-ginza/bin/activate 
$ python ./scripts/identify_speaker.py ./data/result/novel_data.csv　./data/result/novel_data.csv
$ deactivate
```
#### LLM Method
```shell
$ source ./venv-wo-ginza/bin/activate 
$ python ./scripts/identify_speaker.py ./data/result/novel_data.csv　./data/result/novel_data.csv --use_llm
$ deactivate
```
#### Tone Metohd
```shell
$ source ./venv-wo-ginza/bin/activate 
$ python ./scripts/identify_speaker.py ./data/result/novel_data.csv　./data/result/novel_data.csv
$ deactivate
```

> [!NOTE]
> Tone Methodで発話者を特定するためにはRule MethodとLLM Methodで発話者を特定する必要があります．

### Create Corpus
会話グループの作成と発話者特定をした小説を用いて会話コーパスを構築します。
```shell
$ source ./venv-w-ginza/bin/activate 
$ python ./scripts/identify_speaker.py ./data/result/novel_data.csv　./data/result/character.json　./data/result/dialogue_corpus.json　
$ deactivate
```

### Pipeline 
```shell
$ ./scripts/run.sh ./data/sample_novel/ ./data/result/novel_data.csv ./data/result/character.json ./data/result/dialogue_corpus.json
```

## License
本手法はMITライセンスで提供しています．

## Citation
TBU