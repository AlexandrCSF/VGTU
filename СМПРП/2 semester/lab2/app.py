import random
import time

import redis
from flask import Flask, request

app = Flask(__name__)
redis_client = redis.Redis(host="redis", port=6379)

raw_data = "I am studying Redis"
key_for_raw_data = "my_raw_data"
key_for_processed_data = "my_processed_data"
key_for_processed_count = "my_processed_count"
key_for_lock = "my_lock"


def get_key_value(key_name):
    key_value = (redis_client.get(key_name) or b"").decode("utf-8")
    return key_value


def set_key_value(key_name, new_value):
    redis_client.set(key_name, new_value)


def get_char_from_key_value(key_name, number):
    key_value = get_key_value(key_name)
    return key_value[number]


def add_to_key_value(key_name, add_value):
    key_value = get_key_value(key_name)
    redis_client.set(key_name, key_value + add_value)


def is_locked(key_name):
    key_set = redis_client.set(key_name, 1, ex=10, nx=True)
    return not key_set


def delete_key(key_name):
    redis_client.delete(key_name)


@app.route("/")
def show():
    raw_data_value = get_key_value(key_for_raw_data)
    processed_data_value = get_key_value(key_for_processed_data)
    return (
        f"Raw value: {raw_data_value}, <br> Processed value: {processed_data_value}"
    )


@app.route("/init/")
@app.route("/init")
def init():
    set_key_value(key_for_raw_data, raw_data)
    set_key_value(key_for_processed_data, "")
    set_key_value(key_for_processed_count, 0)
    return "OK"


@app.route("/process/")
def process():
    response = ""
    with_lock = bool(request.args.get("with-lock"))
    response += f"with lock: {with_lock}<br>"

    while True:
        if with_lock:
            waiting = ""
            while is_locked(key_for_lock):
                waiting += "-"
                time.sleep(0.01)
            response += f"waiting for lock: {waiting}<br>"

        processed_count = int(get_key_value(key_for_processed_count) or 0)
        if processed_count >= len(raw_data):
            if with_lock:
                delete_key(key_for_lock)
            break

        unprocessed_char = get_char_from_key_value(key_for_raw_data, processed_count)
        processed_char = str(unprocessed_char).upper()
        time.sleep(random.random())

        add_to_key_value(key_for_processed_data, processed_char)
        response += f"changed char: {unprocessed_char} -> {processed_char}<br>"

        set_key_value(key_for_processed_count, processed_count + 1)

        if with_lock:
            delete_key(key_for_lock)

        time.sleep(random.random())

    return response
