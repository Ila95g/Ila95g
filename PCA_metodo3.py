from sklearn.decomposition import PCA
from neo.io.neomatlabio import NeoMatlabIO
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
import argparse
import logging
import os
import pandas as pd
import numpy as np
import pickle as pk
import keras
from keras import Sequential, optimizers
from keras.layers import Dense, LSTM, Dropout, TimeDistributed, Bidirectional
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns






logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:  %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[
        logging.FileHandler("window_split_data.log", mode='w'),
        logging.StreamHandler()
    ]
)

def obj_mapping(id): #mappatura in 7 classi

    if id == 11:
        id = 41
    if id == 13:
        id = 34
    if id == 14:
        id = 44
    if id == 16:
        id = 25
    if id == 12:
        id = 54
    if id == 21 or id == 22 or id == 31 or id == 41 or id == 42 or id == 43:
       return 2 #classe gialla
    if id == 23 or id == 24 or id == 25 or id == 26:
       return 1 #classe azzurra
    if id == 32 or id == 33 or id == 34 or id == 35 or id == 36:
       return 6 #classe rosa
    if id == 44 or id == 45 or id == 46:
       return 5 #classe nera
    if id == 51 or id == 52 or id == 53 or id == 54 or id == 55 or id == 56:
       return 3 #classe blu
    if id == 15 or id == 61 or id == 62 or id == 63 or id == 64 or id == 65 or id == 66: 
       return 4 #classe rossa
    if id == 71 or id == 72 or id == 73 or id == 74 or id == 75 or id == 76:
       return 0 #classe verde
    
    return id


def P_C_A(X_train):
    pca=PCA(n_components=200).fit(X_train)
    pk.dump(pca, open("pca_20_metodo3.pkl","wb"))
    X_reduced=pca.fit_transform(X_train)
    return X_reduced,pca

def dataset_PCA(dirpath,df):
    X = []
    size_matrix_bin=[]
    size_matrix_multi=[]

    file_ids = df['img_id'].to_numpy(dtype=int).tolist()
    filenames = list(map(str, file_ids))
    for idx in range(len(filenames)):
        filepath = os.path.join(dirpath, 'img_'+filenames[idx])
        binned_spk = np.load(filepath)
        np_matrix = binned_spk['arr_0']
        size_matrix_bin.append(np_matrix.shape[1])
        size_matrix_multi.append(np_matrix.shape[1])
        
        X.append(np_matrix.T)

    X_=np.vstack(X)   
    return X_,size_matrix_bin,size_matrix_multi

def prepare_detection_dataset(PCA_Dataset,size_matrix,df,lookback, lookahead):
    X=PCA_Dataset
    X_bin=[]
    y_bin=[]
    
    tt=[]
    y=[]

    file_ids = df['img_id'].to_numpy(dtype=int).tolist()
    filenames = list(map(str, file_ids))
    obj_ids = df['obj_id'].to_numpy(dtype=int)
    hold = df['Hold'].to_numpy(dtype=int)
    rew = df['Rew'].to_numpy(dtype=int)

    for idx in range(len(size_matrix)):
        n=size_matrix.pop(0)
        n_col=np.arange(n)
        np_matrix=X[n_col,:].T
        X=np.delete(X,n_col,axis=0)
        tt.append(np_matrix)
        for i in range(np_matrix.shape[1]-lookback-lookahead):
            t = []
            for j in range(0, lookback):
                matrix=np_matrix[:, [(i+j)]]                
                t.append(matrix)
            X_bin.append(t)
            if (i+lookback+lookahead) >= hold[idx] and (i+lookback+lookahead) < rew[idx]:  # attivazione
                y_bin.append(1)
                y.append(obj_mapping(obj_ids[idx]))
            else: 
                y_bin.append(0)
                y.append(-1)

    data_set_bin = np.array(X_bin)
    y_label_bin = np.array(y_bin)
    y_label = np.array(y)    
    
    label_encoder_bin = LabelEncoder()
    label_encoder_bin.fit(y_label_bin)
    integer_encoded_bin = label_encoder_bin.transform(y_label_bin).reshape(len(y_label_bin), 1)
    onehot_encoder_bin = OneHotEncoder(sparse=False)
    onehot_encoder_bin.fit(integer_encoded_bin)
    label_bin = onehot_encoder_bin.transform(integer_encoded_bin)
    
    label_encoder = LabelEncoder()
    label_encoder.fit(y_label)
    integer_encoded = label_encoder.transform(y_label).reshape(len(y_label), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoder.fit(integer_encoded)
    label = onehot_encoder.transform(integer_encoded)

    return data_set_bin, label_bin

def prepare_classification_dataset(PCA_Dataset,size_matrix,df,lookback, lookahead):
    X=PCA_Dataset
    X_multi=[]    
    y_multi = []
    tt = []
    
    
    file_ids = df['img_id'].to_numpy(dtype=int).tolist()
    filenames = list(map(str, file_ids))
    obj_ids = df['obj_id'].to_numpy(dtype=int)
    hold = df['Hold'].to_numpy(dtype=int)
    rew = df['Rew'].to_numpy(dtype=int)
    

    for idx in range(len(size_matrix)):
        n=size_matrix.pop(0)
        n_col=np.arange(n)
        np_matrix=X[n_col,:].T
        X=np.delete(X,n_col,axis=0)
        tt.append(np_matrix)
        for i in range(np_matrix.shape[1]-lookback-lookahead):
            t = []
            for j in range(0, lookback):
                matrix=np_matrix[:, [(i+j)]]                
                t.append(matrix)
            if (i+lookback+lookahead) >= hold[idx] and (i+lookback+lookahead) < rew[idx]:  # attivazione
                X_multi.append(t)
                y_multi.append(obj_mapping(obj_ids[idx]))
               
    data_set = np.array(X_multi)
    y_label = np.array(y_multi)    
    
    label_encoder = LabelEncoder()
    label_encoder.fit(y_label)
    integer_encoded = label_encoder.transform(y_label).reshape(len(y_label), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoder.fit(integer_encoded)
    label = onehot_encoder.transform(integer_encoded)

    return data_set, label        


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help=".mat file containing brain recordings", default='data/MRec42_40_binned_spiketrains',
                        type=str)
    
    parser.add_argument("-o", "--outdir", help="Output data directory", default='./data',
                        type=str)
    parser.add_argument("--lookback", help="length of the sequence to be classified", default=12,
                        type=int)
    parser.add_argument("--lookahead", help="length of the sequence to be classified", default=0,
                        type=int)
    
    args = parser.parse_args()
    
    data_prefix = os.path.basename(os.path.normpath(args.dataset))
    csv_file = os.path.join(args.dataset,data_prefix+'.csv')

    logging.info('\n')
    logging.info('----------------')
    logging.info('Building windows')
    logging.info(args.dataset)
    logging.info('----------------')
    logging.info('\n')
    logging.info('Loading data...')

    subdirectory = 'lookback_' + str(args.lookback) + '_' + 'lookahead_' + str(args.lookahead) + '_' + 'PCA_Dataset_Metodo3'
    if not os.path.exists(os.path.join(args.outdir,data_prefix,subdirectory)):
        os.makedirs(os.path.join(args.outdir,data_prefix,subdirectory))

    binned_samples_df = pd.read_csv(csv_file)
    hold = binned_samples_df['Hold']
    rew = binned_samples_df['Rew']
    img = binned_samples_df['img_id']
    obj = binned_samples_df['obj_id']
    
    logging.info('Remove under-represented classes')
    obj_to_remove = binned_samples_df.groupby('obj_id').Start.count().loc[lambda p: p < 2]
    for obj_r in obj_to_remove.keys():
        logging.info('Removed object: ', obj_r)
        binned_samples_df = binned_samples_df[binned_samples_df.obj_id!=obj_r]
        
    train_df, test_df = train_test_split(binned_samples_df, test_size=0.8, random_state=42, stratify=binned_samples_df.obj_id)
    
    X_Train,size_matrix_bin_Train,size_matrix_multi_Train=dataset_PCA(args.dataset,train_df)
    X_Test,size_matrix_bin_Test,size_matrix_multi_Test=dataset_PCA(args.dataset,test_df)
    X_data,size_matrix_bin,size_matrix_multi=dataset_PCA(args.dataset,binned_samples_df)
    
    
    print('\nFull dataset dimension:')
    print(X_data.shape)
    
    
    
    logging.info('\n')
    logging.info('Creating PCA (20%) dataset...')  
    Data_PCA_Train,pca = P_C_A(X_Train)
    Data_PCA_Test=pca.transform(X_Test)
    Data_PCA=pca.transform(X_data)
    
    logging.info('\n')
    logging.info('Processing Bin PCA dataset ..')        
        
    X_bin,y_bin=prepare_detection_dataset(Data_PCA, size_matrix_bin, binned_samples_df, args.lookback, args.lookahead)   
    X_train_bin, y_train_bin=prepare_detection_dataset(Data_PCA_Train, size_matrix_bin_Train, train_df , args.lookback, args.lookahead)
    X_test_bin, y_test_bin=prepare_detection_dataset(Data_PCA_Test, size_matrix_bin_Test, test_df , args.lookback, args.lookahead)

        
    logging.info('\n')
    logging.info('Save binary PCA dataset')
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'binary_fullset.npz'), 'bw') as trainfile:
        np.savez(trainfile, X=X_bin, y=y_bin)
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'binary_trainset.npz'), 'bw') as trainfile:
        np.savez(trainfile, X=X_train_bin, y=y_train_bin)
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'binary_testset.npz'), 'bw') as testfile:
        np.savez(testfile, X=X_test_bin, y=y_test_bin)
            
    
    logging.info('\n')
    logging.info('Processing Multi PCA dataset ..')       
    
    X_multi, y_multi =prepare_classification_dataset(Data_PCA, size_matrix_multi, binned_samples_df, args.lookback, args.lookahead)
    X_train_multi,y_train_multi=prepare_classification_dataset(Data_PCA_Train, size_matrix_multi_Train, train_df, args.lookback, args.lookahead)
    X_test_multi,y_test_multi=prepare_classification_dataset(Data_PCA_Test, size_matrix_multi_Test, test_df, args.lookback, args.lookahead)
    
    
    
    logging.info('\n')
    logging.info('Save multiclass PCA dataset')
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'multi_fullset.npz'), 'bw') as trainfile:
        np.savez(trainfile, X=X_multi, y=y_multi)
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'multi_trainset.npz'), 'bw') as trainfile:
        np.savez(trainfile, X=X_train_multi, y=y_train_multi)
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'multi_testset.npz'), 'bw') as testfile:
        np.savez(testfile, X=X_test_multi, y=y_test_multi)
    
    logging.info('\n')
    logging.info('Completed')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Name of BRNN model", default='MRec40_40_binned_spiketrains_lookback_12_lookahead_0_PCA_Dataset_Metodo2_',
                        type=str)
    parser.add_argument("--lookback", help="length of the sequence to be classified", default=12,
                        type=int)
    parser.add_argument("--lookahead", help="length of the sequence to be classified", default=0,
                        type=int)
    
    args = parser.parse_args()

    data_prefix = os.path.basename(os.path.normpath(args.model))
    path_multi=data_prefix+'multi_model'
    
    logging.info('\n')
    logging.info('----------------')
    logging.info('Loading Multi Models')
    model=keras.models.load_model(path_multi)
    logging.info('Multi Model Info')
    model.summary()
    
   
    callback = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=20, restore_best_weights=True)

    
    model.fit(X_train_multi, y_train_multi, epochs=50, batch_size=128,
              validation_data=(X_test_multi,y_test_multi), callbacks=[callback])
    
    model.save(data_prefix+'_MRec40_on_MRec42')
    
    predicted_value = model.predict(X_test_multi)
    predicted_value = predicted_value.argmax(axis=1)
    y_test_multi = y_test_multi.argmax(axis=1)
    
    #confusion matrix
    
    cm2 = confusion_matrix(y_test_multi,predicted_value)
    
    fig, ax = plt.subplots(figsize=(16,14))
    sns.heatmap(cm2, annot=cm2, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('Load_all_classes_matrix_BRNN_PCA.png')
    plt.close()
    print('All classes report')
    print(classification_report(y_test_multi, predicted_value))