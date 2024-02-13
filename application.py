# import all the packages 
from flask import Flask,render_template,request
import pandas as pd 
import numpy as np
import pickle

## Create instance of an application
application = Flask(__name__)
app = application

# show the homepage of your application
@app.route('/')
def homepage():
    return render_template('index.html')

# Predition Logic 
@app.route('/predict',methods=['POST'])
def predict_species():
    if request.method=='GET':
        return render_template('index.html')
    else:
         # load the preprocessor / num pipe line 
        with open('notebook/preprocessor.pkl','rb') as file1:
            pre = pickle.load(file1)
        with open('notebook/model.pkl','rb') as file2:
            model = pickle.load(file2)
        # Get input from form
        sep_len = float(request.form.get('sepal_length'))
        sep_wid = float(request.form.get('sepal_width'))
        pet_len = float(request.form.get('petal_length'))
        pet_wid = float(request.form.get('petal_width'))
        xnew = pd.DataFrame([sep_len,sep_wid,pet_len,pet_wid]).T
        xnew.columns = pre.get_feature_names_out()
        # Preprocess the data 
        xnew_pre = pre.transform(xnew)
        prediction =model.predict(xnew_pre)[0]
        #Probablity
        prob = model.predict_proba(xnew_pre)
        #Max_prob
        prob = round(np.max(prob),4)
        return render_template('index.html',prediction=prediction,prob=prob)

    
#run the application 
if __name__ =='__main__':
    app.run(host='0.0.0.0',debug=True)

