import os
from ftapp import db
from ftapp.models import GlobalVariables, Screener, Regression
import ftapp.core.marketdata_manager as marketdatamanager
import ftapp.core.constants as constants

db.drop_all()

db.create_all()

globals = GlobalVariables(capital=constants.SV, leverage=0, friction=0)

db.session.add(globals)

db.session.commit()

globals_from_db = GlobalVariables.query.all()


screener= Screener(name="Dummy")

db.session.add(screener)

db.session.commit()


regression= Regression(name="Dummy")

db.session.add(regression)

db.session.commit()


# print("Upload market data")
# marketdataManager.upload_all_marketdata()

print('Done!')
