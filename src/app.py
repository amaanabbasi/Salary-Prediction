from flask import Flask, render_template, request, url_for
import pickle
import os
from sklearn.linear_model import LinearRegression
import numpy as np


app = Flask(__name__)

def load_model():
    global model 
    model = pickle.load(open('model.pkl','rb'))
    return model

@app.route("/", methods=['POST','GET'])
def index():
    # Input Form
    context = None
    if request.method == 'POST':
        field1 = request.form['field1']
        field2 = request.form['field2']
        field3 = request.form['field3']
        
        fields = np.array([[field1, field2, field3]])
        fields = fields.astype(np.float64)

        prediction = model.predict(fields)

        context = {
            "average_salary" : prediction[0]
        }

        return render_template('index.html', context=context)
        

    return render_template('index.html', context=context)

load_model()

# if __name__ == "__main__":
    
#     app.run(host=os.getenv('IP', '0.0.0.0'), 
#             port=int(os.getenv('PORT', 8080)))
#     app.run(debug=True)


