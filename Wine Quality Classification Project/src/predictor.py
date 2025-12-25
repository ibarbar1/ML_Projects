import numpy as np




def predict_quality(model, feature_list):
    arr = np.asarray(feature_list).reshape(1, -1)
    prediction = model.predict(arr)[0]
    return "Good Wine" if prediction == 1 else "Bad Wine"