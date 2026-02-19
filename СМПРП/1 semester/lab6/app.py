from os import environ
from flask import Flask, request
from pika import BlockingConnection, URLParameters
from pika.exceptions import AMQPConnectionError

app = Flask(__name__)
worker_number = environ.get('WORKER_NUMBER')
fanout_exchange_name = 'my-fanout-exchange'
my_queue_name_template = 'my-queue-'
my_queue_name = my_queue_name_template + worker_number


def get_rabbitmq_channel():
    rabbitmq_connection_parameters = URLParameters(url='http://rabbitmq/')
    rabbitmq_connection = BlockingConnection(parameters=rabbitmq_connection_parameters)
    rabbitmq_channel = rabbitmq_connection.channel()
    rabbitmq_channel.exchange_declare(exchange=fanout_exchange_name, exchange_type='fanout', durable=False)
    rabbitmq_channel.queue_declare(queue=my_queue_name, durable=False)
    rabbitmq_channel.queue_bind(queue=my_queue_name, exchange=fanout_exchange_name)
    return rabbitmq_channel


@app.route('/')
def show():
    try:
        rabbitmq_channel = get_rabbitmq_channel()
    except AMQPConnectionError:
        return 'RabbitMQ is not ready yet'

    _, _, body = rabbitmq_channel.basic_get(my_queue_name, auto_ack=True)
    if body:
        message = bytes.decode(body, 'utf-8')
        return f'Message is: {str(message)}'
    return 'Queue is empty'


@app.route('/publish/')
def publish():
    try:
        rabbitmq_channel = get_rabbitmq_channel()
    except AMQPConnectionError:
        return 'RabbitMQ is not ready yet'

    message = request.args.get('message')
    reciever = request.args.get('reciever')

    if reciever == 'all':
        rabbitmq_channel.basic_publish(
            exchange=fanout_exchange_name,
            routing_key='',
            body=message.encode('utf-8'),
        )
        return 'OK - all'
    elif reciever in ['0', '1', '2']:
        rabbitmq_channel.basic_publish(
            exchange='',
            routing_key=my_queue_name_template + reciever,
            body=message.encode('utf-8'),
        )
        return f'OK - {reciever}'
    return 'Invalid reciever'