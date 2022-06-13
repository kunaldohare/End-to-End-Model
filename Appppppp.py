import numpy
import pickle

import numpy as np
from flask import Flask, render_template, request, jsonify
import sklearn

app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('index.html')


@app.route ('',methods=['POST','GET'])
def results():
    bedrooms= float (request.form['bedrooms'])
    bathrooms=float (request.form['bathrooms'])
    sqft_living=float (request.form['sqft_living'])
    sqft_lot=float (request.form['sqft_lot'])
    floors=float (request.form['floors'])
    waterfront=float (request.form['waterfront'])
    view=float (request.form['view'])
    conditiion=float (request.form['conditiion'])
    sqft_above=float (request.form['sqft_above'])
    sqft_basement = float(request.form['sqft_basement'])
    yr_built=float (request.form['yr_built'])
    yr_renovated= (request.form['yr_renovated'])

    x=np.array([[bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,conditiion,sqft_above,yr_built,yr_renovated,]])
    model = pickle.load(open('model.pkl','rb'))
    Y_prediction = model.predict(X)
    return jsonify({'Model Prediction': float(Y_prediction)})

if __name__ =='__main__':
    app.run(debug = True, port= 1010)
