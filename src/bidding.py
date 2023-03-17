import json
import os

import h5py
import joblib
from dotenv import load_dotenv
from keras.models import load_model

from utils import create_sequences

load_dotenv()
SEQ_LEN = 14
import pandas as pd


def predict(data_df, features, target, scaler, forecaster):
    scaled_data = pd.DataFrame(scaler.transform(data_df[features + [target]]),
                               columns=features + [target])
    sequenced_scaled_data = create_sequences(scaled_data,
                                             lookback=SEQ_LEN,
                                             univariate=False,
                                             target=target,
                                             inference=True)
    unscaled_forecast = forecaster.predict(sequenced_scaled_data)
    # forecast = scaler.inverse_transform(unscaled_forecast)
    return unscaled_forecast


features = [
    'Impressions',
    'AbsoluteTopImpressionPercentage',
    'TopImpressionPercentage',
    'SearchImpressionShare',
    'SearchTopImpressionShare',
    'SearchRankLostTopImpressionShare',
    'Clicks',
    'Cost_gbp',
    'CpcBid_gbp',
]
features_date = features + ['Date']
target = 'CpcBid_gbp'
features.remove(target)
with open(
        os.path.join(os.environ['MODELS_PATH'], 'kmeans_clustered_dict.json'),
        'r') as f:
    clustered = json.load(f)

latest = max(os.listdir(os.environ['RUNS_PATH']))
for cluster in clustered.keys():
    scaler = joblib.load(
        os.path.join(os.environ['RUNS_PATH'], latest, f'cluster_{cluster}',
                     'scaler.joblib'))
    with open(
            os.path.join(os.environ['RUNS_PATH'], latest, f'cluster_{cluster}',
                         f'cluster_{cluster}.h5'), 'rb') as model_file:
        keras_model = load_model(h5py.File(model_file, 'r'))
