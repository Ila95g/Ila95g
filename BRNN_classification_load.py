import logging
import argparse
import os
import seaborn as sns
from tensorflow import keras
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:  %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[
        logging.FileHandler("window_split_data.log", mode='w'),
        logging.StreamHandler()
    ]
)

def load_data(filepath):
    # load existing data
    windows_dataset = np.load(filepath)
    X = windows_dataset['X']
    # X = X.reshape(-1, X[0].shape[0], X[0].shape[1], 1)
    # resize with 3 channels, but it is still a problem the input dimension
    # X = np.repeat(X, 3, axis=3)
    y = windows_dataset['y']
    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Name of BRNN model", default='MRec40_40_binned_spiketrains_lookback_12_lookahead_0_PCA_Dataset_',
                        type=str)
    parser.add_argument("--lookback", help="length of the sequence to be classified", default=12,
                        type=int)
    parser.add_argument("--lookahead", help="length of the sequence to be classified", default=0,
                        type=int)
    parser.add_argument("-d", "--dataset",help="Name of dataset from previous step, without extension", 
                        default='data/MRec42_40_binned_spiketrains/lookback_12_lookahead_0_PCA_Dataset',
                        type=str)
    
    args = parser.parse_args()
    
    data_prefix = os.path.basename(os.path.normpath(args.model))
    path_multi=data_prefix+'multi_model'
    
    
    logging.info('Caricamento dataset')
    testset_path = os.path.join(args.dataset, 'multi_fullset.npz')
    test_set,label_test=load_data(testset_path)
   
    
    logging.info('\n')
    logging.info('----------------')
    logging.info('Loading Multi Model')
    model=keras.models.load_model(path_multi)
    logging.info('Multi Model Info')
    model.summary()
    
    
    predicted_value = model.predict(test_set)
    predicted_value = predicted_value.argmax(axis=1)
    label_test = label_test.argmax(axis=1)
    
    #confusion matrix
    
    cm2 = confusion_matrix(label_test,predicted_value)
    
    fig, ax = plt.subplots(figsize=(16,14))
    sns.heatmap(cm2, annot=cm2, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('Load_all_classes_matrix_BRNN_PCA.png')
    plt.close()
    print('All classes report')
    print(classification_report(label_test, predicted_value))