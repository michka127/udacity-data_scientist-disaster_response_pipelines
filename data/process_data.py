import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Imports two datasets behind the files paths 
    in the arguments and merges them
    
    Args:
    messages_filepath: path to a .csv file containing the messages for import
    categories_filepath: path to a .csv file containing the cathegories 
    for the messages for import
    
    Returns:
    df: pandas dataframe. Merged data from the arguments
    
    """    
    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)    
    # merge datasets
    df = messages.merge(categories, how='inner', on='id')
    
    return df


def clean_data(df):
    """cleans data 
    Cleaning steps include merging the messages and categories datasets, 
    splitting the categories column into separate, clearly named columns, 
    converting values to binary, and dropping duplicates.
    
    Args:
    df: pandas dataframe. Data that needs cleaning
    
    Returns:
    df: pandas dataframe. Cleaned data    
    """
    # create a dataframe of the individual category columns
    categories = df.categories.str.split(";",expand=True,)
    
    #get category names
    category_colnames = categories.loc[0, :].str.split(pat="-").str[0]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # set each value to be the last character of the string and convert to numeric
    for column in categories:
        categories[column] = pd.to_numeric(categories[column].str[-1])
        
    # drop the original categories column from `df` 
    # and concat with the new `categories` dataframe instead
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # return cleaned dataframe
    return df


def save_data(df, database_filename):
    """stores data into a SQLite database in a specified database file path
    If the database does not exist, it will be created
    
    Args:
    df: pandas dataframe. Clean data, possibly from clean_data()
    database_filename: Path to SQLite database. 
    
    Returns:
    None   
    """
        
    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql('disaster_data', engine, index=False)
      


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()