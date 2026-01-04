uv python pin 3.11

uv venv venv-w-ginza -p 3.11
source venv-w-ginza/bin/activate 
uv pip install -r requirements-w-ginza.txt
deactivate

uv venv venv-wo-ginza -p 3.11
source venv-wo-ginza/bin/activate 
uv pip install -r requirements-wo-ginza.txt
deactivate