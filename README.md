# Bidding engine

The repository contains the notebook and source code for the development and training of the Google Ads keyword bidding engine. The purpose of the engine is to automate the keyword bidding in order to boost an increase in number of sessions given historical data and metrics of the keywords.

## Training data

The dataset used for training of the developed machine learning system comes as a Google Ads report, which can be downloaded with [Google Ads API](https://developers.google.com/google-ads/api/docs/reporting/overview). For example, [Python SDK](https://developers.google.com/google-ads/api/docs/client-libs/python) can be used to access the Google Ads account and to download keyword stats with the [example script](https://github.com/googleads/google-ads-python/blob/main/examples/reporting/get_keyword_stats.py). The data should be than placed into the `./data/raw` in the current directory root. For the development, `bidding_data.feather` should be placed in `./data/raw`

## Inference data

To collect the recent keywords stats:

1. Configure [google-ads.yaml](configs/google-ads.yaml)

    Populate [google-ads.yaml](configs/google-ads.yaml) with `customer_id`, `client_secret`,`refresh_token` and `login_customer_id` tokens.

2. Download `{timestamp}.csv` file

    The `.csv` report contains the performance metrics of campaigns associated with the `customer_id` customer over the last week to `./data/raw`.Run the script below to create `{timestamp}.csv` file, where `{timestamp}` is auto-generated in format `YYYYmmDDHHMMSS`

```
python src/get_keyword_stats_report.py
```

## Prerequisites

Set up the virtual environment and install the required libraries.

```
python -m venv .venv
source ./Scripts/activate
python -m pip install --upgrade pip setuptools
pip install -r config/requirements-dev.txt
```

## Strategy

The general strategy is to learn and replicate the historical strategy by implementing multivariate `Cpc` forecasting with the available keyword performance metrics from Google Ads Reporting. Alternatively, the `Cpc` update can be proposed to be based on the trend of `Impressions` as a proxy for the market price of the keyword.

Due to a number of individual keywords and existence of broad-match modifiers, it is suggested to semantically cluster the keywords using pre-trained embeddings.

### Implementation plan

1. Keywords embedding

    As the number of individual and unique keywords is large (over 4200 including broad match modifiers (BMM) and just under 3900 excluding BMMs), the first step is to conduct semantic keyword clustering using pre-trained embeddings. [Sentence transformer](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v1) language model was used as the intention was to keep semantics of the keyword as a whole.

2. Embeddings clustering

    Performed k-means and agglomerative clustering (beneficial when the number of clusters is unknown and hard to guess)

3. Forecasting

    For each cluster perform multi-variate single-step forecasting of `Cpc`. If the forecast is bigger than the moving average projection and tolerance -> 'push', if smaller -> 'pull', if within the tolerance -> 'zero'


### Development

[`nbs`](./nbs/) directory contains the Jupyter notebooks used to explore the data and prototype. The working Python modules, content of which can be used to build training and inference workflows, are located in [`src`](./src/) directory.
