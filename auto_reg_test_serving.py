import numpy
import pandas
import json
import requests
from linear_input import  *

def prediction():
    x, train_x,test_x = input_data()
    # Wrap bitstring in JSON
    out_pp = numpy.array(train_x[-1])
    strafe = out_pp.astype(numpy.float32)
    #sdrake = dict(strafe)
    data = json.dumps({"inputs": float(strafe)})
    data1 = json.dumps({"inputs": float(1)})
    json_response = requests.post("http://0.0.0.0:8501/v1/models/auto_reg_m:predict", \
                                     data=data1)
    print(json_response)
    response = json.loads(json_response.text)
    print(json_response.status_code)
    predicted_value = response['outputs']
    print(predicted_value)
    return predicted_value

strake = prediction()
print(strake)
