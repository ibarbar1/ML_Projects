from data_loader import load_wine_data
from feature_engineering import preprocess
from exploration import explore, plot_correlations
from model import train_model
from predictor import predict_quality




def main():
    df = load_wine_data()
    explore(df)
    plot_correlations(df)

    df = preprocess(df)
    model, X = train_model(df)

    sample = X.iloc[0].tolist()
    print("\nPrediction for sample:", predict_quality(model, sample))


if __name__ == "__main__":
    main()