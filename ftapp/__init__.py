import logging.handlers
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

app.config['LOG_FILE_NAME'] = 'user_data/trace.log'
#for running on Heroku
#app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
#app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('SQLALCHEMY_DATABASE_URI')
app.config.from_object('config')


db = SQLAlchemy(app)

import ftapp.core.constants as constants

if not os.path.exists(constants.PORTVAL_FOLDER):
    os.makedirs(constants.PORTVAL_FOLDER)
if not os.path.exists(constants.MARKETDATA_FOLDER):
    os.makedirs(constants.MARKETDATA_FOLDER)
if not os.path.exists(constants.USERDATA_FOLDER):
    os.makedirs(constants.USERDATA_FOLDER)
if not os.path.exists(constants.USERINPUT_FOLDER):
    os.makedirs(constants.USERINPUT_FOLDER)
if not os.path.exists(constants.WEIGHTS_FOLDER):
     os.makedirs(constants.WEIGHTS_FOLDER)

BASIC_FORMAT = "%(message)s"
logfilename = app.config.get('LOG_FILE_NAME')
handler = logging.handlers.WatchedFileHandler(os.environ.get('LOGFILE', logfilename))
formatter = logging.Formatter(BASIC_FORMAT)
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
logger.addHandler(handler)


import sys
import logging

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)


from ftapp.main.routes import main
from ftapp.screen.routes import screen
#from ftapp.database.routes import database
from ftapp.optimize.routes import optimization
from ftapp.errors.handlers import errors

app.register_blueprint(main)
app.register_blueprint(screen)
#app.register_blueprint(database)
app.register_blueprint(optimization)
app.register_blueprint(errors)


