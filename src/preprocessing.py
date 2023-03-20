import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from dotenv import load_dotenv


def fill_mode(df, feature: str, return_modes: bool = False):
    """
    The fill_mode function takes a dataframe and a feature as input.
    It then finds all the unique CriterionIds that have missing values for that feature,
    and fills in those missing values with the mode of each respective CriterionId.
    The function returns the filled dataframe.

    Parameters
    ----------
        df
            Pass in the dataframe to be filled
        feature: str
            Specify the column name of the feature to be filled
        return_modes: bool
            Return the modes of each criterion

    Returns
    -------
        A dataframe with the missing values filled in
    """
    filled_df = df.copy(deep=True)
    missing_criterionids = df.loc[df[feature].isnull(), 'CriterionId'].unique()
    modes = []
    for criterion in missing_criterionids:
        mode = filled_df.loc[filled_df['CriterionId'] == criterion,
                             feature].mode().values[0]
        filled_df.loc[filled_df['CriterionId'] == criterion,
                      feature] = filled_df.loc[filled_df['CriterionId'] ==
                                               criterion,
                                               feature].fillna(value=mode)
        modes.append(mode)
    if return_modes: return filled_df, modes
    return filled_df


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """
    The downcast function takes a dataframe and returns a copy of the dataframe with all float columns downcasted to
    the smallest possible dtype that can hold the values in each column.  All integer columns are downcasted to unsigned
    integers.  This function is useful for reducing memory usage when working with large datasets.

    Parameters
    ----------
        df: pd.DataFrame
            Specify the dataframe that will be downcasted

    Returns
    -------
        A new dataframe with the downcasted columns
    """
    dc_df = df.copy(deep=True)
    fcols = dc_df.select_dtypes('float').columns
    icols = dc_df.select_dtypes('integer').columns
    dc_df[fcols] = dc_df[fcols].apply(pd.to_numeric, downcast='float')
    dc_df[icols] = dc_df[icols].apply(pd.to_numeric, downcast='unsigned')
    return dc_df


def clean_rescale_obj_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    The clean_rescale_obj_metrics function takes a dataframe as input and returns a copy of the dataframe with
    the following changes:
        1. All columns containing 'Share' or 'Percentage' in their name are converted to float dtype.
        2. The values in these columns are rescaled between 0 and 100 (or 0 and 1 if they contain 'Share').  This is done by first removing any leading or trailing characters, then subtracting the minimum value from each value, then dividing by the maximum value for that column.

    Parameters
    ----------
        df: pd.DataFrame
            Specify the dataframe that will be used in the function

    Returns
    -------
        A copy of the original dataframe with cleaned and rescaled object columns
    """
    cr_df = df.copy(deep=True)
    obj_metrics_features = [
        c for c in cr_df.select_dtypes('object').columns
        if 'Share' in c or 'Percentage' in c
    ]
    for col in obj_metrics_features:
        cr_df[col] = cr_df[col].str.lstrip('<').str.rstrip('%').astype(float)
        cr_df[col] = (cr_df[col] - cr_df[col].min()) / cr_df[col].max()
        if 'Percentage' in col:
            cr_df[col] *= 100
    return cr_df


def preprocess_raw_history(raw_data_file: str,
                           interim_data_file: str,
                           return_df: bool = False) -> pd.DataFrame:
    """
    The preprocess_raw_history function takes a raw data file and returns an interim data file.

    Parameters
    ----------
        raw_data_file: str
            Specify the name of the raw data file to be used
        interim_data_file: str
            Specify the name of the interim data file
        return_df: bool
            Return the dataframe if true, otherwise it will just save the file

    Returns
    -------
        The interim_df dataframe
    """
    raw_df = pd.read_feather(
        os.path.join(os.environ['RAW_DATA_PATH'], raw_data_file))
    raw_df = raw_df.sort_values(by=['CriterionId', 'Date']).reset_index(
        drop=True)
    interim_df = fill_mode(raw_df, feature='Cost', return_modes=False)
    interim_df = clean_rescale_obj_metrics(interim_df)
    interim_df = downcast(interim_df)
    micros_to_gbp = interim_df['Cost'].div(interim_df['Cost_gbp']).replace({
        np.nan:
        0,
        np.inf:
        0
    }).unique()[1:].mean()
    interim_df['CpcBid_gbp'] = interim_df['CpcBid'] / micros_to_gbp
    interim_df.to_feather(
        os.path.join(os.environ['INTERIM_DATA_PATH'], interim_data_file))
    if return_df: return interim_df


if __name__ == '__main__':
    load_dotenv()
    parser = ArgumentParser()
    parser.add_argument('--raw-history-file',
                        type=str,
                        default='bidding_data.feather')
    parser.add_argument('--interim-data-file',
                        type=str,
                        default='interim_data.feather')
    args = parser.parse_args()
    interim_df = preprocess_raw_history(
        raw_data_file=args.raw_history_file,
        interim_data_file=args.interim_data_file,
        return_df=False)
