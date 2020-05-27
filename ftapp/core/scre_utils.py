import os
from flask import flash
import ftapp.core.utils as utils
from werkzeug.utils import secure_filename
import ftapp.core.constants as constants

def get_factor_analysis(file,ycol, discretize, cat_cols, num_cols, name ):
    filename = secure_filename(file.filename)
    file.save(os.path.join('.',constants.USERDATA_FOLDER, filename))
    flash('Your screening factors have been uploaded.', 'success')
    # load data
    data = utils.load_screener_data(filename)
    # split into x and y
    if ycol:
        y = data[ycol]
        x = data.copy()
        x.drop([y], axis=1)
    else:
        # default to last column for y
        y = data.iloc[:, -1]
        x = data.copy()
        x.drop(x.columns[-1], axis=1)

    if discretize:
        y = utils.discretize_y(x, y)

    x = utils.clean_data(x, cat_cols, num_cols)
    # fill_nas
    x = x.fillna(x.mean())
    # save cleaned data for further use
    utils.save(x, y, name)
    # graph and display
    eigen_plot_url = utils.eigenvalues_plt(x)
    corr_plot_url = utils.corr_matrix(x)
    data_table = x.describe()
    return eigen_plot_url, corr_plot_url, data_table