import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from dotenv import load_dotenv
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction import text
from umap import UMAP
from wordcloud import WordCloud


def visualize_topics(kwds: list,
                     topic_model: BERTopic,
                     ext: str = 'png') -> None:
    """
    The visualize_topics function takes a list of keywords and a trained topic model as input.
    It then creates word clouds for each topic in the model, using the keywords that are assigned to that topic.
    The function also saves two images: one showing the hierarchy of topics in the model, and another showing
    the distribution of documents across topics.

    Parameters
    ----------
        kwds: list
            Specify the keywords to be used for the topic model
        topic_model: BERTopic
            Pass the model to the function
        ext: str
            Image file extension

    Returns
    -------
        None
    """
    kwds_topic_df = topic_model.get_document_info(
        kwds).loc[:, ['Document', 'Topic', 'Name']]
    if os.path.isdir(os.environ['IMAGES_PATH']):
        for f in os.listdir(os.environ['IMAGES_PATH']):
            os.remove(os.path.join(os.environ['IMAGES_PATH'], f))
    os.makedirs(os.environ['IMAGES_PATH'], exist_ok=True)
    for topic in sorted(kwds_topic_df['Topic'].unique()):
        wordcloud = WordCloud().generate(
            kwds_topic_df.loc[kwds_topic_df['Topic'] == topic,
                              'Document'].str.cat(sep=' '))
        name = kwds_topic_df.loc[kwds_topic_df['Topic'] == topic,
                                 'Name'].unique()[0]
        plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(name)
        plt.savefig(os.path.join(os.environ['IMAGES_PATH'], f'{name}.{ext}'),
                    bbox_inches='tight')
    topic_model.visualize_hierarchy().write_image(
        os.path.join(os.environ['IMAGES_PATH'], f'hierarchy.{ext}'))
    topic_model.visualize_barchart().write_image(
        os.path.join(os.environ['IMAGES_PATH'], f'barchart.{ext}'))
    embeddings = topic_model.embedding_model.embedding_model.encode(kwds)
    reduced_embeddings = UMAP(n_neighbors=15,
                              n_components=2,
                              min_dist=0.0,
                              metric='cosine').fit_transform(embeddings)
    topic_model.visualize_documents(
        kwds, reduced_embeddings=reduced_embeddings).write_image(
            os.path.join(os.environ['IMAGES_PATH'], f'clusters_2d.{ext}'))
    topic_model.visualize_heatmap().write_image(
        os.path.join(os.environ['IMAGES_PATH'],
                     f'topics_similarity_matrix.{ext}'))


def save_topics_info(kwds: list, topic_model: BERTopic) -> None:
    """
    The save_topics_info
 function saves the topic model, a json file containing the topics,
    a csv file containing inextion about each document's topic distribution and a csv
    file containing inextion about each topic. The function takes two arguments:
    kwds (list) - A list of keywords to be used in the BERTopic model.
    topic_model (BERTopic) - A BERTopic object.

    Parameters
    ----------
        kwds: list
            Specify the keywords that are to be used in the topic model
        topic_model: BERTopic
            Save the topic model to a file

    Returns
    -------
        None
    """
    topic_model.save(os.path.join(os.environ['TOPICS_PATH'], 'topic_model'))
    with open(os.path.join(os.environ['TOPICS_PATH'], 'top_words.json'),
              'w') as f:
        json.dump(topic_model.get_topics(), f)
    with open(os.path.join(os.environ['TOPICS_PATH'], 'topic_labels.json'),
              'w') as f:
        json.dump(topic_model.topic_labels_, f)
    with open(os.path.join(os.environ['TOPICS_PATH'], 'repr_kwds.json'),
              'w') as f:
        json.dump(topic_model.representative_docs_, f)
    topic_model.get_document_info(kwds).to_csv(
        os.path.join(os.environ['INTERIM_DATA_PATH'], 'all_kwds.csv'))
    topic_model.get_topic_info().to_csv(
        os.path.join(os.environ['INTERIM_DATA_PATH'], 'topics_info.csv'))
    topic_model.get_document_info(kwds).loc[:, ['Document', 'Name']].groupby(
        by='Name').agg(lambda x: ' '.join(x)).to_csv(
            os.path.join(os.environ['INTERIM_DATA_PATH'], 'topics.csv'))


def extract_topics(interim_data_file: str) -> None:
    """
    The extract_topics function takes a single argument, the name of an interim data file.
    The function reads in the interim data file and extracts all unique criteria from it.
    It then creates a BERTopic model using default parameters for each step in the pipeline:
        Step 1 - Extract embeddings: SentenceTransformer
        Step 2 - Reduce dimensionality: UMAP
        Step 3 - Cluster reduced embeddings: HDBSCAN
        Step 4 - Tokenize topics: CountVectorzer
        Step 5 - Extract topic words: c-TF-IDF

    Parameters
    ----------
        interim_data_file: str
            Specify the file name of the interim data file to be used in this function

    Returns
    -------
        None
    """
    interim_df = pd.read_feather(
        os.path.join(os.environ['INTERIM_DATA_PATH'], interim_data_file))
    kwds = interim_df['Criteria'].unique().tolist()
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    umap_model = UMAP(n_neighbors=15,
                      n_components=10,
                      min_dist=0.0,
                      metric='cosine')
    hdbscan_model = HDBSCAN(min_cluster_size=100,
                            metric='euclidean',
                            cluster_selection_method='eom',
                            prediction_data=True)
    stop_words = text.ENGLISH_STOP_WORDS.union(['holiday', 'holidays'])
    vectorizer_model = text.CountVectorizer(stop_words=list(stop_words))
    ctfidf_model = ClassTfidfTransformer()
    topic_model = BERTopic(
        embedding_model=embedding_model,  # Step 1 - Extract embeddings
        umap_model=umap_model,  # Step 2 - Reduce dimensionality
        hdbscan_model=hdbscan_model,  # Step 3 - Cluster reduced embeddings
        vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
        ctfidf_model=ctfidf_model,  # Step 5 - Extract topic words
    )
    topics, probs = topic_model.fit_transform(kwds)
    save_topics_info(kwds, topic_model)
    visualize_topics(kwds, topic_model)
    kwds_df = topic_model.get_document_info(kwds)
    interim_df['Topic'] = interim_df['Criteria'].map(
        lambda x: kwds_df.loc[kwds_df['Document'] == x, 'Topic'].item())
    interim_df['Topic_name'] = interim_df['Criteria'].map(
        lambda x: kwds_df.loc[kwds_df['Document'] == x, 'Name'].item())
    save_processed_data(interim_df, topic_model)


def save_processed_data(interim_df, topic_model):
    """
    The save_processed_data function takes the interim dataframe and topic model as inputs.
    It then creates a copy of the interim dataframe, loads in the forecast features and target from JSON files,
    and appends them to a list called features_date. It then removes all files from PROCESSED_DATA_PATH using os.system().
    Next it loops through each topic in topic model's keys (topic labels) and creates a new dictionary entry for each one with its name as key.
    The value is set to be processed df filtered by that specific topic label, grouped by date summing up all values for that day across

    Parameters
    ----------
        interim_df
            Pass the dataframe containing the interim data
        topic_model
            Identify the topic labels

    Returns
    -------
        None
    """
    processed_df = interim_df.copy(deep=True)
    with open(
            os.path.join(os.environ['FEATURES_PATH'],
                         'forecast_features.json'), 'r') as f:
        features = json.load(f)
    with open(os.path.join(os.environ['FEATURES_PATH'], 'target.json'),
              'r') as f:
        target = json.load(f)
    features_date = features + ['Date', target]
    if os.path.isdir(os.environ['PROCESSED_DATA_PATH']):
        for f in os.listdir(os.environ['PROCESSED_DATA_PATH']):
            os.remove(os.path.join(os.environ['PROCESSED_DATA_PATH'], f))
    os.makedirs(os.environ['PROCESSED_DATA_PATH'], exist_ok=True)
    forecast_data = {}
    for topic in topic_model.topic_labels_.keys():
        forecast_data[topic] = processed_df.loc[processed_df['Topic'] == topic,
                                                features_date].groupby(
                                                    by='Date').sum()
        forecast_data[topic].to_csv(
            os.path.join(os.environ['PROCESSED_DATA_PATH'],
                         f'processed_{topic}.csv'))


if __name__ == '__main__':
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--interim-data-file',
                        type=str,
                        default='interim_data.feather')
    args = parser.parse_args()
    extract_topics(interim_data_file=args.interim_data_file)
