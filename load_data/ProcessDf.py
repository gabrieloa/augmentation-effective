import pandas as pd
from .PreProcess import process_regex


def process_df(df, text_column: str,
               label_column: str) -> pd.DataFrame:
    """Process and save dataframe

    :param path_df: path to the dataframe.
    :param delimiter: delimiter of the dataframe.
    :param text_column: name of the column with the text.
    :param label_column: name of the column with the label.
    :return: Dataframe with the processed text and the label column.

    """
    df = df[~df[text_column].isnull()]
    df = df[~df[label_column].isnull()]

    df['text_process'] = df[text_column].apply(
        lambda x: process_regex(x))

    df = df.rename(columns={label_column: 'Label'})

    return df[[text_column, 'text_process', 'Label']]