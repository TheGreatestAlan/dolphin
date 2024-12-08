import os

from flask import Flask

app = Flask(__name__)

# Ollama API Base URL
OLLAMA_API_BASE_URL = "http://localhost:11434"

if __name__ == '__main__':
    os.environ['FLASK_SKIP_DOTENV'] = 'true'
    app.run(debug=True, host=os.environ['REST_ADDRESS'], port=os.environ['REST_PORT'], use_reloader=False)
