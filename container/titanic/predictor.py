# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
from io import StringIO
import sys
import signal
import traceback
import h5py
import flask
import h5py
from keras.models import load_model
from sklearn import preprocessing

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import pandas as pd
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)



graph=tf.get_default_graph()
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

clf = load_model(os.path.join(model_path, 'keras.h5'),compile=True)


class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    # @classmethod
    # def get_model(cls):
    #     """Get the model object for this instance, loading it if it's not already loaded."""
    #     if cls.model == None:
    #         #with open(os.path.join(model_path, 'keras.h5'), 'r') as inp:
    #         #custom_objects={'CRF': CRF,'crf_loss':crf_loss,'crf_viterbi_accuracy':crf_viterbi_accuracy}
    #         cls.model = load_model(os.path.join(model_path, 'keras.h5'))
    #     return cls.model

    @classmethod
    def predict(cls, test_data):
    	
        #cat=['Credit_History','Education','Gender']
        #test_data.columns=cat
        global clf,graph

        test_data=pd.DataFrame(test_data)
        headers = test_data.iloc[0]
        test_data  = pd.DataFrame(test_data.values[1:], columns=headers)

        test_data['Title'] = test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        test_data['Family_members'] = test_data['SibSp'] + test_data['Parch']
        test_data.drop(["Name","SibSp","Parch","Cabin","Ticket","PassengerId"],axis=1,inplace=True)
        test_data["Title"]=test_data["Title"].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')
        test_data["Title"]=test_data["Title"].replace(['Mlle','Ms'],'Miss')
        test_data["Title"]=test_data["Title"].replace(['Mme'],'Mrs')

        test_data["Fare"]=test_data["Fare"].astype(float)
        test_data["Age"]=test_data["Age"].astype(float)
        test_data["Pclass"]=test_data["Pclass"].astype(int)
   
        test_data['Age'] = test_data['Age'].fillna(test_data.groupby(["Title"])['Age'].transform('mean'))
        test_data['Fare'] = test_data['Fare'].fillna(test_data.groupby(["Pclass"])['Fare'].transform('mean'))
        test_data['Embarked']=test_data['Embarked'].fillna("S")
  
        test_data.loc[(test_data['Pclass'] == 1),'Pclass_Band'] = 3
        test_data.loc[(test_data['Pclass'] == 3),'Pclass_Band'] = 1
        test_data.loc[(test_data['Pclass'] == 2),'Pclass_Band'] = 2
        

        test_data = pd.get_dummies(test_data,columns=["Sex","Embarked","Title"])
        test_data.drop(["Sex_male","Embarked_S"],axis=1,inplace=True)
        test_data["Title_Mr"]=0
        test_data["Pclass_Band"]=test_data["Pclass_Band"].astype(int)
        test_data.drop(["Pclass"],axis=1,inplace=True)

        with graph.as_default() as graph:
            prediction = clf.predict_classes(test_data)
        return prediction


# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = StringIO(data)
        data = pd.read_csv(s, header=None)

    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.shape[0]))

    # Do the prediction
    predictions = ScoringService.predict(data)
    print(predictions)
    print(type(predictions))
    predictions=predictions.reshape(len(predictions))
    # Convert from numpy back to CSV
    out = StringIO()
    pd.DataFrame({'results':predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()
    #result=predictions
    return flask.Response(response=result, status=200, mimetype='text/csv')
