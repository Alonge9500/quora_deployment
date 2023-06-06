from flask import Flask,render_template,request,redirect,url_for
from flask_wtf import CSRFProtect
from forms import PredictForm
import pickle
import os
import numpy as np
from .preprocess import feature

model_path = os.path.join(os.path.dirname(__file__), '..', 'pickles', 'model.pkl')

with open(model_path,'rb')as y:
    model = pickle.load(y)


app = Flask(__name__)
app.secret_key = b'flyftfrhefguygdduylhutfrtfk'
csrf = CSRFProtect(app)


@app.route("/", methods=['GET','POST'])
def home():
    
    form = PredictForm()
    if form.validate_on_submit():
        question1 = form.question1.data
        question2 = form.question2.data
        
        
        
        
        
        
        prediction=model.predict(data)[0]
        
        print(prediction)
        return redirect(url_for('result',prediction=prediction))
        
    else:
        return render_template('home.html',form=form)


@app.route("/result")
def result():
    prediction = (request.args.get('prediction')).split(' ')
    pred_yes = float(prediction[0][1:]) * 100
    pred_no = float(prediction[1][:-2]) * 100
    
    prediction = [pred_yes,pred_no]
    print(prediction)
    return render_template('result.html', pred=prediction)

if __name__ == "__main__":
    app.run(debug=True)
    
    