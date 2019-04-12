from flask import render_template, flash, redirect, url_for, request
from app import app, db
from app.forms import LoginForm, RegistrationForm, EditProfileForm, ResetPasswordRequestForm, ResetPasswordForm, InputData
from flask_login import current_user, login_user, logout_user, login_required
from app.models import User
from werkzeug.urls import url_parse
from datetime import datetime
from app.email import send_password_reset_email
from app.helper_functions import *
from app.simulation import FF_Simulation


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():

    form = InputData()
    if form.validate_on_submit():
        #input = form.topred.data
        #pred = random_forest.fit_model(input)
        #flash('Your prediction is {}'.format(pred))

        pts_dict = {}
        pts_dict['QB'] = [0.04, 4, 0.1, 6, 0, 0]
        pts_dict['RB'] = [0.3, 0.1, 0.5, 6]
        pts_dict['WR'] = [0.1, 0.5, 6]
        pts_dict['TE'] = [0.1, 0.5, 6]

        FF_Simulation(db_name='app.db', set_year=2018, pts_dict=pts_dict)

        flash('Your data is pulled')
        return redirect(url_for('index'))

    # return the index template with the PostForm and items of pagination
    return render_template('index.html', title='Home', form=form)






#==========
# Register and Login
#==========

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route('/user/<username>')
@login_required
def user(username):

    # pull the current user
    user = User.query.filter_by(username=username).first_or_404()

    # return the user.html page
    return render_template('user.html', user=user)

@app.before_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.utcnow()
        db.session.commit()

@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm(current_user.username)
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.about_me = form.about_me.data
        db.session.commit()
        flash('Your changes have been saved.')
        return redirect(url_for('edit_profile'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.about_me.data = current_user.about_me
    return render_template('edit_profile.html', title='Edit Profile', form=form)


#=========
# Password Reset
#=========

@app.route('/reset_password_request', methods=['GET', 'POST'])
def reset_password_request():

    # if user is signed in, redirect to index home page
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    # otherwise, load the password reset form
    form = ResetPasswordRequestForm()

    # upon submission, pull the email information and send the request email
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            send_password_reset_email(user)
        flash('Check your email for the instructions to reset your password')
        return redirect(url_for('login'))

    # return the password reset template to allow submission
    return render_template('reset_password_request.html', title='Reset Password', form=form)


@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):

    # if the user is already logged in, return to home page
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    # otherwise, verify that the token sent to user is correct
    user = User.verify_reset_password_token(token)
    if not user:
        return redirect(url_for('index'))

    # if everything checks out, go to the reset password form
    form = ResetPasswordForm()
    if form.validate_on_submit():

        # commit the new password and send to login page afterwards
        user.set_password(form.password.data)
        db.session.commit()
        flash('Your password has been reset.')
        return redirect(url_for('login'))

    return render_template('reset_password.html', form=form)