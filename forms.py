import os
from flask_wtf import FlaskForm  
from wtforms import validators, ValidationError  
from wtforms import SubmitField,TextAreaField
import pickle




class PredictForm(FlaskForm):
    question1 = TextAreaField('Question 1', validators=[validators.DataRequired()])
    question2 = TextAreaField('Question 2', validators=[validators.DataRequired()])
    
    submit = SubmitField('Submit')


    
    
    
     
    
    
    
    