source ./venv-w-ginza/bin/activate
python ./scripts/create_corpus_format.py $1 $2
echo "Create corpus format!"
python ./scripts/create_character_name_list.py $2 $3
echo "Create corpus chracter list!"
python ./scripts/assign_dialogue_group.py $2 $2
echo "Assign dialogue group!"
python ./scripts/identify_speaker.py $2 $3 $2 --use_ginza
echo "Identify speaker by the rule-base method!"
deactivate

source ./venv-wo-ginza/bin/activate
python ./scripts/identify_speaker.py $2 $3 $2  --use_llm
python ./scripts/identify_speaker.py $2 $3 $2
echo "Speaker identification has been completed!!"
python ./scripts/create_dialogue_corpus.py $2 $3 $4
echo "Dialogue Corpus Construction  has been completed!!"
deactivate
