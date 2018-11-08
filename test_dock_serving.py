from linear_input import  *
from auto_reg_test_serving import *

x, train_x,test_x = input_data()
strake = prediction(test_x)
print(strake)
