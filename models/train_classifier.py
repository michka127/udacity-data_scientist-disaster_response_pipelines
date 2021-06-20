import sys
from sqlalchemy import create_engine

import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    """ load data for training the model
    
    Args:
    database_filepath: string. path to a SQLite database 
    containing the data for training the model
    
    Returns:
    X: pandas dataframe. Data to predict on for classification
    Y: pandas dataframe. True Classifiers for data in X
    category_names: category names for the classifiers in Y 
    """
    # load data from database
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table('disaster_data', engine) 
    
    #adjust data for function return values
    X = df['message']
    Y = df.drop(['id','message', 'original','genre'],axis=1)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    """ custom tokenize function using nltk to case normalize, 
    lemmatize, and tokenize text
    """
    # replace URLs with a placeholder
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # tokenize words
    tokens = word_tokenize(text)
    
    # clean tokens
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ Setting up the machine learning pipeline to vectorize,
    then apply TF-IDF to the text and predict classifiers
    
    Args:
    None
    
    Returns:
    model: Sklearn machine learning pipeline. Model to predict classifiers
    """
    # set up machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),        
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
         ])
    
    # define parameters for model optimization
    parameters = {'vect__max_df': (0.5, 1.0),
        'vect__max_features': (None, 5000)}
    
    #set up model
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):    
    """ Find the best parameters for the model and output f1 score, 
    precision and recall for the test set  for each category.
    
    Args:
    model: Sklearn machine learning pipeline. Model to predict classifiers
    X_test: pandas dataframe. Test data to predict on for classification
    Y_test: pandas dataframe. True Classifiers for test data in X_test
    category_names: list. List of classifier names
    
    Returns:
    None
    """
    # predict labels for test data
    Y_pred = model.predict(X_test)
    
    # print best parameter set
    print("\nBest Parameters:", model.best_params_)
    
    # print model performance    
    for col in range(Y_pred.shape[1]):
        print('Category: ',category_names[col])
        print(classification_report(Y_test.values[:,col], Y_pred[:,col]))
        

def save_model(model, model_filepath):
    """ Find the best parameters for the model and output f1 score, 
    precision and recall for the test set  for each category.
    
    Args:
    model: Sklearn machine learning pipeline. Model to predict classifiers
    model_filepath: String. Path to the location where the model shall be saved
    
    Returns:
    None
    """
    # save model to path
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
        

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()