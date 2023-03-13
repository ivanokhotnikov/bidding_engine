# Bidding engine

The repository contains the notebook and source code for the development and training of the Google Ads keyword bidding engine. The purpose of the engine is to automate the keyword bidding in order to boost an increase in number of sessions given historical data and metrics of the keywords.

## Dataset

The dataset used for training of the developed machine learning system comes as a Google Ads report, which can be downloaded with [Google Ads API](https://developers.google.com/google-ads/api/docs/reporting/overview). For example, [Python SDK](https://developers.google.com/google-ads/api/docs/client-libs/python) can be used to access the Google Ads account and to download keyword stats with the [example script](https://github.com/googleads/google-ads-python/blob/main/examples/reporting/get_keyword_stats.py). The data should be than placed into the `./data/raw` in the current directory root. For the development, `bidding_data.feather` should be placed in `./data/raw`
