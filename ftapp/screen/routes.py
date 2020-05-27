from flask import Blueprint, render_template, flash, redirect, url_for, request
#from ftapp import db
from ftapp.screen.forms import ScreenerForm, RegressionVariables
import ftapp.core.constants as constants
from ftapp.models import Screener, Regression
from ftapp import db
import pandas as pd

from sklearn.decomposition import PCA

from flask import Flask,abort,render_template,request,redirect,url_for

app = Flask(__name__)
UPLOAD_FOLDER = './user_data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
import ftapp.core.scre_utils as scre_utils
import ftapp.core.utils as utils
import os
screen= Blueprint('screen', __name__)

@screen.route('/screen/file_upload/',methods = ['GET','POST'])
def upload_file():
    form = ScreenerForm()
    #indicator ID

    if request.method =='POST':
        #get file, save file
        screener_name = form.name.data
        screener = Screener(name=screener_name)
        db.session.add(screener)
        db.session.commit()

        file = request.files['file']

        ycol = form.ycol.data
        discretize = form.discretize.data
        cat_cols = form.cat_cols.data
        num_cols = form.num_cols.data
        name = form.name.data

        if file:
            eigen_plot_url, corr_plot_url, data_table = scre_utils.get_factor_analysis(file,ycol, discretize, cat_cols, num_cols, name )
            #flash('See below for your initial factor analysis.', 'success')
            return render_template('screener_data.html', title = 'Data Visualization',name = screener_name, graph_e= eigen_plot_url, graph_c = corr_plot_url, table = data_table.to_html())

    return render_template('file_upload.html', form = form)

@screen.route('/screen/run_model',methods = ['GET','POST'],  strict_slashes=False)
def run_model():
    form = RegressionVariables()
    screeners = Screener.query.all()

    if request.method =='POST':
        #get the dataset to use
        screener_id = int(request.form.getlist("fortest")[0])
        filename= screeners[screener_id-1]
        #get all the variables
        model_name = form.name.data
        model = Regression(name=model_name)
        db.session.add(model)
        db.session.commit()

        #load data and split
        X= pd.read_csv('./user_data/{}-x.csv'.format(filename), index_col = 0)
        y = pd.read_csv('./user_data/{}-y.csv'.format(filename), index_col = 0)

        if form.train_test_split.data:
            split = form.train_test_split.data
        else:
            split = 0.2

        X_train, X_test, y_train, y_test, idx_train, idx_test = utils.split_data(X, y, split)

        if form.pca_b.data:
            pca = PCA()
            if form.pca_d.data:
                #calc num pc
                summary = utils.pca_summary(pca, X)
                summary.sdev**2
            elif form.pca_num.data:
                num_pc = form.pca_num.data
            pca = PCA(num_pc)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        model = form.model.data
        #print(model)
        score, y_pred, clf = utils.evaluate(model, X_train, X_test, y_train, y_test, optimize =form.optimize.data )
        graph_c_url = utils.confusion_matrix(clf, X_test, y_test)
        classes = [-1,0,1]
        graph_cp_url = utils.classificationreport(clf, classes, X_train, y_train, X_test, y_test)

        increase, stay, decrease = utils.get_stocks_classification(y_pred, idx_test)


        #return redirect(url_for('screen.screen_results'))
        return render_template('screen_result.html', title='Model Output', name=model_name, \
                               graph_c = graph_c_url, graph_cp = graph_cp_url, up = increase, down = decrease,\
                               total_s = len(y), test = len(y_test), train = len(y_train),\
                               drop = len(decrease), inc = len(increase), same = stay)

    return render_template('screen.html', form = form, screeners = screeners)

if __name__ == '__main__':
    app.run(debug = True)

