"""Flaskサーバへの要求
"""
import requests

def query():
    r = requests.post('http://localhost:8000/hello')
    response = r.content

    print("Server response: %s in client" % response)

if __name__ == "__main__":
    print("Starting client...")
    query()