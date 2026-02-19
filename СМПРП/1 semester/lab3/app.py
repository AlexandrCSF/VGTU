import redis
from flask import Flask

app = Flask(__name__)
my_key = 'MY_KEY'
redis_client = redis.Redis(host='redis', port=6379)


def get_key_value(key_name):
    key_value = redis_client.get(key_name) or 0
    return int(key_value)


def change_key_value(key_name, value):
    current_key_value = get_key_value(key_name)
    redis_client.set(key_name, current_key_value + value)


@app.route('/')
def home():
    current_key_value = get_key_value(my_key)
    return f'Current value: {current_key_value}'


@app.route('/incr/')
def incr():
    change_key_value(my_key, 1)
    return 'Incremented'


@app.route('/decr/')  # аналогичным образом уменьшаем значение ключа на
def decr():
    change_key_value(my_key, -1)
    return 'Decremented'
