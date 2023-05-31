from pymongo import MongoClient

mongo = MongoClient("mongodb://localhost:27017/")
db = mongo["CritiCat"]


change_stream = db.match.watch()

for change in change_stream:
    print(change)