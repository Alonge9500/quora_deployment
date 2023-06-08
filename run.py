from flask import Flask,render_template,request,redirect,url_for
from flask_wtf import CSRFProtect
from forms import PredictForm
import pickle
import os
import numpy as np
from feature_extract import final_features

model_path = os.path.join(os.path.dirname(__file__), 'pickles', 'model.pkl')

with open(model_path,'rb')as y:
    model = pickle.load(y)


app = Flask(__name__)
app.secret_key = b'bjdhsybuenngstuwbhdsku'
csrf = CSRFProtect(app)


@app.route("/", methods=['GET','POST'])
def home():
    
    form = PredictForm()
    if form.validate_on_submit():
        question1 = form.question1.data
        question2 = form.question2.data
        
        data = final_features(question1,question2)
        
        
        
        
        
        prediction=model.predict(data)[0]
        
        print(prediction)
        if prediction == 1:
            result = 'Questions is Duplicate'
        elif prediction == 0:
            result = 'Questions not Duplicate'
        else:
            result = prediction
        return redirect(url_for('result',prediction=result))
        
    else:
        return render_template('home.html',form=form)


@app.route("/result")
def result():
    prediction = request.args.get('prediction')

    return render_template('result.html', pred=prediction)

if __name__ == "__main__":
    app.run(debug=True)
    
    