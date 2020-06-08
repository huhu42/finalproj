from flask_wtf import FlaskForm
from wtforms import IntegerField, FloatField, StringField, BooleanField, SubmitField, SelectField
from flask_wtf.file import FileField, FileAllowed
from wtforms.validators import DataRequired, Length, ValidationError, NumberRange
from ftapp.models import Screener, Regression


class ScreenerForm(FlaskForm):
    name = StringField('Data Name', validators=[DataRequired(), Length(min=2, max=30)])
    #screeninput = FileField('Screener Data File (.csv)', validators=[FileAllowed(['csv'])])
    ycol = StringField('Predict Column', validators=[Length(min=2, max=30)])
    discretize = BooleanField('Discretize Predict')
    #fromfolder = BooleanField('From folder')
    #submit = SubmitField('Upload')
    cat_cols =StringField('Catagorical Columns', validators=[Length(min=2, max=1000)])
    num_cols = StringField('Numerical Columns', validators=[Length(min=2, max=1000)])

    def validate_name(self, name):
        indicator = Screener.query.filter_by(name=name.data).first()
        if indicator:
            raise ValidationError('That indicator name is taken! Please choose a different one.')


class RegressionVariables(FlaskForm):
    name = StringField('Screener Name', validators=[DataRequired(), Length(min=2, max=30)])

    pca_b = BooleanField('Apply PCA')
    pca_num = IntegerField('Number of Components to Use', validators=[NumberRange(1, 1000)])
    pca_d = BooleanField('Use Default (PCs with Standard deviation > 1)')
    optimize = BooleanField('Optimize Model with GridSearch')
    train_test_split = FloatField('Test Split (decimal, default 0.2)', validators=[ NumberRange(0.00001, 1)])

    model = SelectField(
        u'Machine Learning Model',
        choices=[(1,'Decision Tree'),(2,'Ada Boost'), (3,'Neural Network'), (4,'Random Forest')]
    )

    submit = SubmitField('Run')

    def validate_name(self, name):
        indicator = Regression.query.filter_by(name=name.data).first()
        if indicator:
            raise ValidationError('That indicator name is taken! Please choose a different one.')

