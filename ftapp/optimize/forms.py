from flask_wtf import FlaskForm
from wtforms import IntegerField, StringField, SubmitField, BooleanField
from wtforms.validators import DataRequired, NumberRange,Length, Required
from wtforms.fields.html5 import DateField


class RequiredIf(Required):
    # a validator which makes a field required if
    # another field is set and has a truthy value

    def __init__(self, other_field_name, *args, **kwargs):
        self.other_field_name = other_field_name
        super(RequiredIf, self).__init__(*args, **kwargs)

    def __call__(self, form, field):
        other_field = form._fields.get(self.other_field_name)
        if other_field is None:
            raise Exception('no field named "%s" in form' % self.other_field_name)
        if bool(other_field.data):
            super(RequiredIf, self).__call__(form, field)


class OptimizeForm(FlaskForm):
    ticker = StringField('Ticker', validators=[DataRequired(), Length(min=1, max=4)])
    start = DateField('Start Date', format='%Y-%m-%d')
    end = DateField('End Date', format='%Y-%m-%d')
    lockup = BooleanField('IPO Lockup Expiration')
    days = IntegerField('Number of Days for Testing (if selecting IPO Lockup Expiration)', validators=[NumberRange(-100, 100)])
    pretrain = BooleanField('Use Pretrained Best Model')
    submit = SubmitField('Run')
