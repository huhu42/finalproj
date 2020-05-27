from datetime import datetime
from ftapp import db

class GlobalVariables(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    capital = db.Column(db.Integer,  nullable=False)
    leverage = db.Column(db.Integer,  nullable=False, default=0)
    long_leverage = db.Column(db.Integer,  nullable=False, default=0)
    short_leverage = db.Column(db.Integer,  nullable=False, default=0)
    friction = db.Column(db.Integer,  nullable=False, default=0)

    def __repr__(self):
        return f"GlobalVariable('{self.capital}', '{self.leverage}', '{self.friction}')"


class Screener(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(30), unique=True, nullable=False)


    def __repr__(self):
        return f"{self.name}"

class Regression(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(30), unique=True, nullable=False)

    def __repr__(self):
        return f"{self.name}"


####--------------#########################not implemented tet