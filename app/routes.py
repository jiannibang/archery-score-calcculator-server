from app import app
from app import getCircle


@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"