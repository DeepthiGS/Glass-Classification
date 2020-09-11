import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask , render_template,request
import  pickle




app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods =['POST'])
def predict():
    sample = pd.read_csv('glass.csv')
    ss= StandardScaler()
    ss.fit_transform(sample.iloc[:,:-1])
    z=[]
    for x in request.form.values():
        z.append(float(x))
    val = ss.transform([np.array(z)])
    prediction = model.predict(val)
    return  render_template('index.html',prediction_text =f'The glass produced with this combination of elements is {prediction} ,which can be mapped according to the notation provided below.')



if __name__== "__main__" :
    app.run(debug=True)


