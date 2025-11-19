
import pickle
import numpy as np
from flask import Flask
from flask import request
from flask import jsonify

with open('salary.bin','rb') as f_in:
    dv,rf =  pickle.load(f_in)

app = Flask('salary')
@app.route('/predict',methods=['POST'])

def predict():
    employee = request.get_json()
    x = dv.transform([employee])
    sa =  np.expm1(rf.predict(x))
    result = {
        'salary':int(sa)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=9696)
    