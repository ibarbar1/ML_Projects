import numpy as np

def predict_object(model, scaler, input_features):
    arr = np.asarray(input_features).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    prediction = model.predict(arr_scaled)[0]

    return "Rock" if prediction == "R" else "Mine"
