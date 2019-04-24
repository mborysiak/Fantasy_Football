from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField, PasswordField, BooleanField, SubmitField, FloatField, IntegerField
from wtforms.validators import ValidationError, DataRequired, Length, Email, EqualTo, NumberRange
from app.models import User

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    password2 = PasswordField(
        'Repeat Password', validators=[DataRequired(), EqualTo('password')]
    )
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Please use a different username.')

class EditProfileForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    about_me = TextAreaField('About me', validators=[Length(min=0, max=140)])
    submit = SubmitField('Submit')

    def __init__(self, original_username, *args, **kwargs):
        super(EditProfileForm, self).__init__(*args, **kwargs)
        self.original_username = original_username

    def validate_username(self, username):
        if username.data != self.original_username:
            user = User.query.filter_by(username=self.username.data).first()
            if user is not None:
                raise ValidationError('Please use a different username.')

class ResetPasswordRequestForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Request Password Reset')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('Password', validators=[DataRequired()])
    password2 = PasswordField(
        'Repeat Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Request Password Reset')

class InputData(FlaskForm):

    # qb specific points
    pass_yds = FloatField('Points Per Pass Yard',
                          default=0.04,
                          validators=[NumberRange(min=0, max=1, message='Enter a number between 0 and 1')])
    pass_tds = FloatField('Points Per Pass TD',
                          default=4,
                          validators=[NumberRange(min=0, max=20, message='Enter a number between 0 and 20')])
    pass_int = FloatField('Points Per Int',
                          default=-2,
                          validators=[NumberRange(min=-5, max=0, message='Enter a number between -5 and 0')])
    pass_sacks = FloatField('Points Per Sack',
                            default=0,
                            validators=[NumberRange(min=-5, max=0, message='Enter a number between -5 and 0')])

    # rushing and receiving stats
    rush_rec_yds = FloatField('Points Per Rush / Receiving Yard',
                              default=0.1,
                              validators=[NumberRange(min=0, max=1, message='Enter a number between 0 and 1')])
    rush_rec_tds = FloatField('Points Per Rush / Receiving TD',
                              default=6,
                              validators=[NumberRange(min=0, max=20, message='Enter a number between 0 and 20')])
    rec_pts = FloatField('Points Per Reception',
                         default=0.5,
                         validators=[NumberRange(min=0, max=5, message='Enter a number between 0 and 5')])


    submit = SubmitField('Submit')


class SimulationData(FlaskForm):

    player_drop = StringField('Picked Player')
    salary_drop = IntegerField('Salaries of Picked Players')

    my_players = StringField('Your Chosen Players')
    my_salaries = IntegerField('Salary of Your Chosen Players')

    iterations = IntegerField('Number of Model Iterations',
                              default=1000,
                              validators=[NumberRange(min=10, max=1000, message='Enter a number between 10 and 1000')])

    submit = SubmitField('Begin Simulation!')

