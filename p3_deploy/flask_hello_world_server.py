"""軽量サーバーライブラリFlaskサーバ
"""

from flask import Flask
app = Flask(__name__)

@app.route("/hello", methods=["POST"])
def hello():
    print("Hello World! in server.")
    return "Hello World!"

if __name__ == "__main__":
    print("Starting flask server...")
    app.run(host='127.0.0.1', port=8000)