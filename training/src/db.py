from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Вставь сюда свою строку подключения
MONGO_URI = "mongodb+srv://dianakhutorna_db_user:1MkZZDOie0bztsUk@yom.adk31s4.mongodb.net/?appName=YOM"

# Подключение к базе данных
try:
    client = MongoClient(MONGO_URI)
    db = client["mydatabase"]
    collection = db["results"]
    print("MongoDB подключена успешно!")
except ConnectionFailure:
    print("Не удалось подключиться к MongoDB.")

def save_result(result: dict):
    """
    Сохраняет словарь в MongoDB и возвращает id записи.
    """
    insert_result = collection.insert_one(result)
    return insert_result.inserted_id