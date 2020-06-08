import ftapp.core.utils as utils
import ftapp.core.constants as constants
#import os
#dir = os.path.dirname(__file__)

def get_train_graph(pretrain, learner, ticker, start_train, end_train, model ):
    if pretrain:
        learner.num_state = constants.NUM_STATES
        best_model = model()
        file_name = ('./{}/{}.h5'.format(constants.WEIGHTS_FOLDER, ticker))
        best_model.load(file_name)
        print("loaded best mode for pretained")
        # TODO: on error display error stock not found
        learner.learner = best_model
        df_trades_train = learner.testPolicy(symbol=ticker, sd=start_train, ed=end_train, sv=constants.SV)
    else:
        df_trades_train = learner.addEvidence(symbol=ticker, sd=start_train, ed=end_train, sv=constants.SV)
    df_trades_train = df_trades_train.assign(Symbol=ticker)

    portvals, benchmark = utils.generate_data(df_trades_train, ticker, start_train, end_train, constants.SV,\
                                                  constants.COMMISSION, constants.IMPACT,"test")
    traingraph_url = utils.create_graph(portvals, benchmark, df_trades_train, 'Training', save=True, show=True)

    return traingraph_url, learner

def get_test_graph(learner, ticker, start_test, end_test):
    df_trades_test = learner.testPolicy(symbol=ticker, sd=start_test, ed=end_test, sv=constants.SV)
    df_trades_test = df_trades_test.assign(Symbol=ticker)
    portvals_test, benchmark_te = utils.generate_data(df_trades_test, ticker, start_test, end_test, constants.SV,
                                                      constants.COMMISSION,
                                                      constants.IMPACT, 'test')
    testgraph_url = utils.create_graph(portvals_test, benchmark_te, df_trades_test, 'Testing', save=True, show=True)
    return testgraph_url