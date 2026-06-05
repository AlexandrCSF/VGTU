import os

from bson.objectid import ObjectId
from flask import Flask, request
from pymongo import MongoClient
import json

app = Flask(__name__)

MONGO_USER = os.getenv('MONGO_ROOT_USER', 'admin')
MONGO_PASSWORD = os.getenv('MONGO_ROOT_PASSWORD', 'pass')
MONGO_HOST = os.getenv('MONGO_HOST', 'mongo')
MONGO_PORT = os.getenv('MONGO_PORT', '27017')

mongo_client = MongoClient(f'mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/')

@app.route('/')
def home():
    return '''
    <h1>Лабораторная работа 5 - Работа с MongoDB из Python</h1>
    <h2>Тематика: АВТОМОБИЛИ</h2>
    <p>База данных: <b>car_dealership</b>, коллекция: <b>cars</b></p>

    <h3>Доступные команды:</h3>
    <pre>
    <b>1. INSERT - Добавление автомобиля:</b>
    /insert/?brand=Toyota&model=Celica&year=1999&price=5000&engine_volume=1.8&color=red&condition=used

    <b>2. INSERT MANY - Добавление нескольких автомобилей:</b>
    /insert-many/

    <b>3. SEARCH - Поиск по любому полю:</b>
    /search/?key=brand&value=Audi
    /search/?key=year&value=2021
    /search/?key=condition&value=new

    <b>4. UPDATE - Обновление по ID:</b>
    /update/?id=66183d136c799b69af12140b&key=year&value=2022

    <b>5. UPDATE BY FIELD - Обновление по полю:</b>
    /update-by-field/?key=brand&value=Toyota&update_key=year&update_value=2021

    <b>6. DELETE - Удаление по полю:</b>
    /delete/?key=brand&value=Toyota

    <b>7. DELETE BY ID - Удаление по ID:</b>
    /delete-by-id/?id=66183d136c799b69af12140b

    <b>8. GET ALL - Показать все автомобили:</b>
    /get-all/

    <b>9. GET STATS - Статистика по коллекции:</b>
    /get-stats/
    </pre>

    <p><i>После выполнения команд проверьте результат в MongoExpress: http://localhost:8081</i></p>
    '''

@app.route('/insert/')
def insert():
    try:
        db_name = request.args.get('db_name', 'car_dealership')
        collection_name = request.args.get('collection_name', 'cars')

        document = {}
        for key, value in request.args.items():
            if key not in ['db_name', 'collection_name']:
                if key in ['year', 'price']:
                    document[key] = int(value)
                elif key == 'engine_volume':
                    document[key] = float(value)
                else:
                    document[key] = value

        if not document:
            return "Ошибка: не указаны поля для добавления", 400

        result = mongo_client[db_name][collection_name].insert_one(document)
        return f"Документ успешно добавлен! ID: {result.inserted_id}"

    except Exception as e:
        return f"Ошибка: {str(e)}", 500

@app.route('/insert-many/')
def insert_many():
    try:
        db_name = request.args.get('db_name', 'car_dealership')
        collection_name = request.args.get('collection_name', 'cars')

        cars = [
            {"brand": "Audi", "model": "A4", "year": 2021, "price": 35000, "engine_volume": 2.0, "color": "black", "condition": "new"},
            {"brand": "Audi", "model": "Q5", "year": 2022, "price": 45000, "engine_volume": 2.0, "color": "white", "condition": "new"},
            {"brand": "BMW", "model": "X5", "year": 2020, "price": 55000, "engine_volume": 3.0, "color": "blue", "condition": "used"},
            {"brand": "Mercedes", "model": "E-class", "year": 2021, "price": 60000, "engine_volume": 2.5, "color": "silver", "condition": "new"},
            {"brand": "Toyota", "model": "Camry", "year": 2019, "price": 25000, "engine_volume": 2.5, "color": "red", "condition": "used"},
            {"brand": "Honda", "model": "Civic", "year": 2020, "price": 22000, "engine_volume": 1.8, "color": "gray", "condition": "used"},
            {"brand": "Audi", "model": "RS6", "year": 2023, "price": 120000, "engine_volume": 4.0, "color": "green", "condition": "new"},
            {"brand": "Tesla", "model": "Model 3", "year": 2022, "price": 50000, "engine_volume": 0, "color": "white", "condition": "new"},
        ]

        result = mongo_client[db_name][collection_name].insert_many(cars)
        return f"Добавлено {len(result.inserted_ids)} автомобилей! ID: {result.inserted_ids}"

    except Exception as e:
        return f"Ошибка: {str(e)}", 500

@app.route('/search/')
def search():
    try:
        db_name = request.args.get('db_name', 'car_dealership')
        collection_name = request.args.get('collection_name', 'cars')
        key = request.args.get('key')
        value = request.args.get('value')

        if not key or not value:
            return "Ошибка: необходимо указать key и value", 400

        if key in ['year', 'price']:
            try:
                value = int(value)
            except ValueError:
                pass
        elif key == 'engine_volume':
            try:
                value = float(value)
            except ValueError:
                pass

        found_documents = mongo_client[db_name][collection_name].find({key: value})

        response = f'<h3>Результаты поиска: {key} = {value}</h3><pre>'
        count = 0
        for document in found_documents:
            response += f'{json.dumps(document, default=str, ensure_ascii=False, indent=2)}\n\n'
            count += 1

        response += f'\nВсего найдено: {count}</pre>'

        if count == 0:
            response = f'<h3>По запросу "{key} = {value}" ничего не найдено</h3>'

        return response

    except Exception as e:
        return f"Ошибка: {str(e)}", 500

@app.route('/update/')
def update():
    try:
        db_name = request.args.get('db_name', 'car_dealership')
        collection_name = request.args.get('collection_name', 'cars')
        document_id = request.args.get('id')
        key = request.args.get('key')
        value = request.args.get('value')

        if not document_id or not key or not value:
            return "Ошибка: необходимо указать id, key и value", 400

        if key in ['year', 'price']:
            try:
                value = int(value)
            except ValueError:
                pass
        elif key == 'engine_volume':
            try:
                value = float(value)
            except ValueError:
                pass

        result = mongo_client[db_name][collection_name].update_one(
            {'_id': ObjectId(document_id)},
            {'$set': {key: value}}
        )

        if result.matched_count > 0:
            return f"Обновлено! Совпадений: {result.matched_count}, Изменено: {result.modified_count}"
        else:
            return f"Документ с ID {document_id} не найден", 404

    except Exception as e:
        return f"Ошибка: {str(e)}", 500

@app.route('/update-by-field/')
def update_by_field():
    try:
        db_name = request.args.get('db_name', 'car_dealership')
        collection_name = request.args.get('collection_name', 'cars')
        key = request.args.get('key')
        value = request.args.get('value')
        update_key = request.args.get('update_key')
        update_value = request.args.get('update_value')

        if not all([key, value, update_key, update_value]):
            return "Ошибка: необходимо указать key, value, update_key, update_value", 400

        if update_key in ['year', 'price']:
            try:
                update_value = int(update_value)
            except ValueError:
                pass

        result = mongo_client[db_name][collection_name].update_many(
            {key: value},
            {'$set': {update_key: update_value}}
        )

        return f"Обновлено! Совпадений: {result.matched_count}, Изменено: {result.modified_count}"

    except Exception as e:
        return f"Ошибка: {str(e)}", 500

@app.route('/delete/')
def delete():
    try:
        db_name = request.args.get('db_name', 'car_dealership')
        collection_name = request.args.get('collection_name', 'cars')
        key = request.args.get('key')
        value = request.args.get('value')

        if not key or not value:
            return "Ошибка: необходимо указать key и value", 400

        result = mongo_client[db_name][collection_name].delete_many({key: value})

        return f"Удалено документов: {result.deleted_count}"

    except Exception as e:
        return f"Ошибка: {str(e)}", 500

@app.route('/delete-by-id/')
def delete_by_id():
    try:
        db_name = request.args.get('db_name', 'car_dealership')
        collection_name = request.args.get('collection_name', 'cars')
        document_id = request.args.get('id')

        if not document_id:
            return "Ошибка: необходимо указать id", 400

        result = mongo_client[db_name][collection_name].delete_one({'_id': ObjectId(document_id)})

        if result.deleted_count > 0:
            return f"Удален документ с ID {document_id}"
        else:
            return f"Документ с ID {document_id} не найден", 404

    except Exception as e:
        return f"Ошибка: {str(e)}", 500

@app.route('/get-all/')
def get_all():
    try:
        db_name = request.args.get('db_name', 'car_dealership')
        collection_name = request.args.get('collection_name', 'cars')

        documents = mongo_client[db_name][collection_name].find()

        response = f'<h3>Все автомобили в коллекции "{collection_name}"</h3><pre>'
        count = 0
        for document in documents:
            response += f'{json.dumps(document, default=str, ensure_ascii=False, indent=2)}\n\n'
            count += 1

        response += f'\nВсего автомобилей: {count}</pre>'
        return response

    except Exception as e:
        return f"Ошибка: {str(e)}", 500

@app.route('/get-stats/')
def get_stats():
    try:
        db_name = request.args.get('db_name', 'car_dealership')
        collection_name = request.args.get('collection_name', 'cars')

        collection = mongo_client[db_name][collection_name]

        total_count = collection.count_documents({})
        audi_count = collection.count_documents({"brand": "Audi"})
        bmw_count = collection.count_documents({"brand": "BMW"})
        new_cars = collection.count_documents({"condition": "new"})
        used_cars = collection.count_documents({"condition": "used"})

        pipeline = [{"$group": {"_id": None, "avg_price": {"$avg": "$price"}}}]
        avg_price_result = list(collection.aggregate(pipeline))
        avg_price = avg_price_result[0]["avg_price"] if avg_price_result else 0

        brand_pipeline = [
            {"$group": {"_id": "$brand", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        brand_stats = list(collection.aggregate(brand_pipeline))

        response = f'''
        <h3>Статистика коллекции "{collection_name}"</h3>
        <pre>
        Всего автомобилей: {total_count}
        Audi: {audi_count}
        BMW: {bmw_count}
        Новые автомобили: {new_cars}
        Подержанные автомобили: {used_cars}
        Средняя цена: ${avg_price:,.2f}

        Распределение по маркам:
        '''
        for stat in brand_stats:
            response += f"  {stat['_id']}: {stat['count']} шт.\n"

        response += '</pre>'
        return response

    except Exception as e:
        return f"Ошибка: {str(e)}", 500

@app.route('/find-audi/')
def find_audi():
    return search_helper('brand', 'Audi')

@app.route('/find-year/')
def find_year():
    year = request.args.get('year', '2021')
    return search_helper('year', year)

def search_helper(key, value):
    try:
        db_name = 'car_dealership'
        collection_name = 'cars'

        if key in ['year', 'price']:
            try:
                value = int(value)
            except ValueError:
                pass

        found_documents = mongo_client[db_name][collection_name].find({key: value})

        response = f'<h3>Результаты поиска: {key} = {value}</h3><pre>'
        count = 0
        for document in found_documents:
            response += f'{json.dumps(document, default=str, ensure_ascii=False, indent=2)}\n\n'
            count += 1

        response += f'\nВсего найдено: {count}</pre>'
        return response

    except Exception as e:
        return f"Ошибка: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)