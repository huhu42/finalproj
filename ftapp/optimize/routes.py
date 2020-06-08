from flask import Blueprint, render_template, request
from ftapp.optimize.forms import OptimizeForm
from ftapp.core.marketsim import Learner_Strat
import ftapp.core.learner as nn
import ftapp.core.constants as constants
import ftapp.core.utils as utils
import datetime as dt
from datetime import timedelta
import ftapp.core.opt_utils as opt_utils

optimization = Blueprint('optimization', __name__)

@optimization.route('/optimization/optimize', methods=['GET', 'POST'])
def optimize():
    #get the form
    form = OptimizeForm()
    #indicators = Indicator.query.all()
    if request.method == 'POST':
        ticker = form.ticker.data
        lockup = form.lockup.data
        days = form.days.data
        pretrain = form.pretrain.data

        if days == None:
            t = 10
        else:
            t = days//2

        if lockup:
            start_train, lockup_date = utils.get_IPO_dates(ticker)
        else:
            startdate = form.start.data.strftime(constants.DATE_FORMAT)
            #lockup expiration
            enddate = form.end.data.strftime(constants.DATE_FORMAT)
            #TODO: variable to optimize for

            #convert to datetime
            start_train = dt.datetime.strptime(startdate, constants.DATE_FORMAT)
            lockup_date = dt.datetime.strptime(enddate, constants.DATE_FORMAT)

        end_train = lockup_date - timedelta(t)
        start_test = end_train
        end_test = lockup_date + timedelta(t)
        #timestamp = time.strftime('%Y%m%d%H%M')

        #TODO: download data if not already available
        #check if already downloaded
        #mdm.download_marketdata(ticker, start_train, end_test)

        #TODO: on error display error stock not found

        #initializa learner
        learner = Learner_Strat(verbose=False, impact=constants.IMPACT)
        #train
        traingraph_url, learner = opt_utils.get_train_graph(pretrain,learner, ticker, start_train, end_train, nn.DQLearner)
        #test
        testgraph_url = opt_utils.get_test_graph(learner, ticker, start_test, end_test)

        #TODO: model selection, validation, get stats, optimization

        return render_template('test_result.html', title='Optimization Result',  train_graph=traingraph_url, test_graph=testgraph_url, ticker = ticker, \
                               start_train = start_train.strftime('%Y-%m-%d') , end_train = end_train.strftime('%Y-%m-%d') , start_test = start_test.strftime('%Y-%m-%d') , end_test = end_test.strftime('%Y-%m-%d') )

    return render_template('optimize.html', title='Opmization Result', form=form)

