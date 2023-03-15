# Bidding engine

The repository contains the notebook and source code for the development and training of the Google Ads keyword bidding engine. The purpose of the engine is to automate the keyword bidding in order to boost an increase in number of sessions given historical data and metrics of the keywords.

## Dataset

The dataset used for training of the developed machine learning system comes as a Google Ads report, which can be downloaded with [Google Ads API](https://developers.google.com/google-ads/api/docs/reporting/overview). For example, [Python SDK](https://developers.google.com/google-ads/api/docs/client-libs/python) can be used to access the Google Ads account and to download keyword stats with the [example script](https://github.com/googleads/google-ads-python/blob/main/examples/reporting/get_keyword_stats.py). The data should be than placed into the `./data/raw` in the current directory root. For the development, `bidding_data.feather` should be placed in `./data/raw`

## Strategy

The Cpc update is proposed to be based on the trend of `Impressions` as a proxy for the market price of the keyword.

### Implementation plan

1. Keywords embedding

    As the number of individual and unique keywords is large (over 4200 including broad match modifiers (BMM) and just under 3900 excluding BMMs), the first step is to conduct semantic keyword clustering using pre-trained embeddings. [Sentence transformer](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v1) language model was used as the intention was to keep semantics of un-tokenized keyword as a whole.

2. Embeddings clustering

    Performed k-means and agglomerative clustering (beneficial when the number of clusters is unknown and hard to guess)

3. Forecasting

    For each cluster perform univariate single-step forecasting of `Impressions`. If the forecast is bigger than the moving average projection and tolerance -> 'push', if smaller -> 'pull', if within the tolerance -> 'zero'
