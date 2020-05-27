import logging.handlers
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
#app.config.from_object('config')

app.config['SECRET_KEY'] = os.environ['SECRET_KEY']
app.config['LOG_FILE_NAME'] = os.environ['LOG_FILE_NAME']
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['SQLALCHEMY_DATABASE_URI']

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
handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", logfilename))
formatter = logging.Formatter(BASIC_FORMAT)
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
logger.addHandler(handler)


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


