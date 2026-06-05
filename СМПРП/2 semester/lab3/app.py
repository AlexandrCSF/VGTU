import random
import threading
import time
import requests
from flask import Flask
import redis

app = Flask(__name__)
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

cache_key_name = 'cache'
update_lock_key_name = 'update_lock'
results_key_name = 'results'
job_count_key_name = 'job_count'

def imitate_job():
    redis_client.incr(job_count_key_name)
    time_to_sleep = random.random()
    time.sleep(time_to_sleep)
    return 'result of work'

def make_request(url):
    start = time.monotonic()
    requests.get(url)
    finish = time.monotonic()
    result = round(finish - start, 6)
    redis_client.rpush(results_key_name, result)
    return result

def make_requests(count, url):
    for _ in range(count):
        make_request(url)

def is_free(key_name):
    key_set = redis_client.set(key_name, 1, nx=True)
    return key_set

@app.route('/without-cache/')
def without_cache():
    result = imitate_job()
    return result

@app.route('/with-cache/')
def with_cache():
    result = redis_client.get(cache_key_name)
    if not result:
        result = imitate_job()
        redis_client.set(cache_key_name, result)
    return result

@app.route('/with-cache-fixed/')
def with_cache_fixed():
    result = redis_client.get(cache_key_name)
    if not result:
        can_update = is_free(update_lock_key_name)
        if can_update:
            result = imitate_job()
            redis_client.set(cache_key_name, result)
            redis_client.delete(update_lock_key_name)
        else:
            time_to_sleep = 0.02
            while not result and time_to_sleep < 5:
                time.sleep(time_to_sleep)
                result = redis_client.get(cache_key_name)
                time_to_sleep *= 2
            if not result:
                result = imitate_job()
                redis_client.set(cache_key_name, result)
    return result

@app.route('/clear/')
def clear():
    redis_client.delete(cache_key_name)
    redis_client.delete(update_lock_key_name)
    redis_client.delete(results_key_name)
    redis_client.delete(job_count_key_name)
    return 'OK'

@app.route('/get-results/')
def get_results():
    job_count = redis_client.get(job_count_key_name)
    results = redis_client.lrange(results_key_name, 0, -1)
    return f'job_count = {job_count}, results = {results}'

@app.route('/without-cache-results/')
def without_cache_results():
    redis_client.delete(results_key_name)
    make_requests(10, 'http://web1:5000/without-cache/')
    return 'OK'

@app.route('/with-cache-results/')
def with_cache_results():
    redis_client.delete(results_key_name)
    redis_client.set(job_count_key_name, 0)
    make_requests(10, 'http://web1:5000/with-cache/')
    return 'OK'

@app.route('/storm-miss-results/')
def storm_miss_results():
    redis_client.delete(results_key_name)
    redis_client.set(job_count_key_name, 0)
    threads = []
    for _ in range(100):
        t = threading.Thread(target=make_request, args=['http://web1:5000/with-cache/'])
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    return 'OK'

@app.route('/storm-miss-fixed-results/')
def storm_miss_fixed_results():
    redis_client.delete(results_key_name)
    redis_client.set(job_count_key_name, 0)
    threads = []
    for _ in range(100):
        t = threading.Thread(target=make_request, args=['http://web1:5000/with-cache-fixed/'])
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)