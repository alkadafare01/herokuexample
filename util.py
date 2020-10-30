import pickle as pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns
    global __locations

    with open("./artifacts/columns.json", "rb") as f:
        print("open")
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
       # print(__locations)# first 3 columns are sqft, bath, bhk

    global __model
 #   f = open("./artifacts/banglore_home_prices_model.pickle", "rb")
 #   classifier = pickle.load(f)
 #   print("load")
 #   f.close()

    with open('./artifacts/banglore_home_prices_model.pickle', 'rb') as f:
        print("open")
        __model = pickle.load(f)
        print(__model)
    print("loading saved artifacts...done")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2)) # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location