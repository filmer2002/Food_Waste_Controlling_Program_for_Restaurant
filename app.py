import flask
import sklearn
import pickle
import pandas as pd

with open(f'model/FWCR_model', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
    
    if flask.request.method == 'POST':
        temperature = flask.request.form['temperature']
        humidity = flask.request.form['humidity']
        rainchance = flask.request.form['rainchance']

        input_variables = pd.DataFrame([[temperature, humidity, rainchance]],
                                       columns=['temperature', 'humidity', 'rainchance'],
                                       dtype=float,
                                       index=['input'])

        prediction = model.predict(input_variables)[0]
    
        return flask.render_template('index.html',
                                     original_input={'Temperature':temperature,
                                                     'Humidity':humidity,                          
                                                     'Chance of Rain':rainchance},
                                     result=('%.2f'%(prediction)),
                                     chicken=('%.2f'%(prediction*0.10)),
                                     pork=('%.2f'%(prediction*0.12)),
                                     fish=('%.2f'%(prediction*0.08)),
                                     vegetable=('%.2f'%(prediction*0.07))
                                     )

if __name__ == '__main__':
    app.run()