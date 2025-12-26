from data_loader import load_sonar_data
from exploration import explore
from feature_engineering import preprocess
from model import train_model
from predictor import predict_object

def main():
    df = load_sonar_data()
    explore(df)

    X, y, scaler = preprocess(df)
    model = train_model(X, y)

    sample = df.drop(columns=["label"]).iloc[0].tolist()
    print("\nPrediction for sample:", predict_object(model, scaler, sample))

if __name__ == "__main__":
    main()