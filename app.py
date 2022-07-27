from urllib.request import Request
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':

        with open('model.pkl', 'rb') as r:
            model = pickle.load(r)

        clump_thickness = float(request.form['clump_thickness'])
        uniform_cell_size = float(request.form['uniform_cell_size'])
        uniform_cell_shape = float(request.form['uniform_cell_shape'])
        marginal_adhesion = float(request.form['marginal_adhesion'])
        single_epithelial_size = float(request.form['single_epithelial_size'])
        bare_nuclei = float(request.form['bare_nuclei'])
        bland_chromatin = float(request.form['bland_chromatin'])
        normal_nucleoli = float(request.form['normal_nucleoli'])
        mitoses = float(request.form['mitoses'])

        datas = np.array((clump_thickness, uniform_cell_size, uniform_cell_shape,
        marginal_adhesion, single_epithelial_size, bare_nuclei,
        bland_chromatin, normal_nucleoli, mitoses))
        datas = np.reshape(datas, (1, -1))

        hasil_prediksi = model.predict(datas)

        return render_template('hasil.html', finalData=hasil_prediksi)
    else:
        return render_template('index.html')

if __name__ == "__main__":
  app.run(debug=True)