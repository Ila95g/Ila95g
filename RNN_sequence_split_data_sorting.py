import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# import matplotlib.pyplot as plt
import pickle
import argparse
import logging

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

def prepare_detection_dataset(dirpath,df,lookback, lookahead,channel_sorting):
    X_bin = []
    y_bin = []
    y = []

    file_ids = df['img_id'].to_numpy(dtype=int).tolist()
    filenames = list(map(str, file_ids))
    obj_ids = df['obj_id'].to_numpy(dtype=int)
    hold = df['Hold'].to_numpy(dtype=int)
    rew = df['Rew'].to_numpy(dtype=int)

    for idx in range(len(filenames)):
        filepath = os.path.join(dirpath, 'img_'+filenames[idx])
        binned_spk = np.load(filepath)
        np_matrix = binned_spk['arr_0']
        np_matrix=np_matrix[channel_sorting,:]
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

def prepare_classification_dataset(dirpath,df,lookback, lookahead, channel_sorting):
    X = []
    y = []

    file_ids = df['img_id'].to_numpy(dtype=int).tolist()
    filenames = list(map(str, file_ids))
    obj_ids = df['obj_id'].to_numpy(dtype=int)
    hold = df['Hold'].to_numpy(dtype=int)
    rew = df['Rew'].to_numpy(dtype=int)

    for idx in range(len(filenames)):
        filepath = os.path.join(dirpath, 'img_'+filenames[idx])
        binned_spk = np.load(filepath)
        np_matrix = binned_spk['arr_0']
        np_matrix=np_matrix[channel_sorting,:]
        for i in range(np_matrix.shape[1]-lookback-lookahead):
            t = []
            for j in range(0, lookback):
                matrix=np_matrix[:, [(i+j)]]                
                t.append(matrix)
            if (i+lookback+lookahead) >= hold[idx] and (i+lookback+lookahead) < rew[idx]:  # attivazione
                X.append(t)
                y.append(obj_mapping(obj_ids[idx]))
               
    data_set = np.array(X)
    y_label = np.array(y)    
    
    label_encoder = LabelEncoder()
    label_encoder.fit(y_label)
    integer_encoded = label_encoder.transform(y_label).reshape(len(y_label), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoder.fit(integer_encoded)
    label = onehot_encoder.transform(integer_encoded)

    return data_set, label
       
        

def read_matrix(line, dirpath):
    global  X_4_sorting
    filepath = os.path.join(dirpath, 'img_'+str(int(line.img_id)))
    binned_spk = np.load(filepath)
    np_matrix = binned_spk['arr_0']
    if len(X_4_sorting)==0:
        X_4_sorting=np_matrix.sum(axis=1)
    else:
        X_4_sorting += np_matrix.sum(axis=1)

def compute_sorting(df_train_4_sorting, dirpath):
    global X_4_sorting
    X_4_sorting = np.array([])
    df_train_4_sorting.apply(lambda line: read_matrix(line, dirpath), axis=1)
    # sorting of the channel by firing rate
    channel_order=np.argsort(X_4_sorting)
    channel_order_descending = channel_order[::-1]
    return channel_order_descending

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Name of dataset from previous step, without extension", default='data/MRec40_40_binned_spiketrains',
                        type=str)
    parser.add_argument("-o", "--outdir", help="Output data directory", default='./data',
                        type=str)
    parser.add_argument("--lookback", help="length of the sequence to be classified", default=12,
                        type=int)
    parser.add_argument("--lookahead", help="length of the sequence to be classified", default=0,
                        type=int)
    # parser.add_argument("--ref-start-event", help="Reference start event for data window definition and grasp labelling", default='Hold',
    #                     type=str)
    # parser.add_argument("--ref-end-event", help="Reference end event for data window definition and grasp labelling", default='Rew',
    #                     type=str)

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
 
    subdirectory = 'lookback_' + str(args.lookback) + '_' + 'lookahead_' + str(args.lookahead) + '_Sorted'
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
    
    train_df, test_df = train_test_split(binned_samples_df, test_size=0.2, random_state=42, stratify=binned_samples_df.obj_id)    
     
     # choose a training set on which compute the sorted order of the channels
    
    logging.info('Sorting channels')
    channel_sorting = compute_sorting(train_df, args.dataset)
    n_channel_40 = 552 #number channel of Mrec40
    channel_sorting=channel_sorting[0:n_channel_40]
    
    logging.info('Processing Binary Sorted dataset')
    X_bin, y_bin = prepare_detection_dataset(args.dataset, binned_samples_df, args.lookback, args.lookahead, channel_sorting)
    X_train_bin, y_train_bin = prepare_detection_dataset(args.dataset, train_df, args.lookback, args.lookahead, channel_sorting)
    X_test_bin, y_test_bin = prepare_detection_dataset(args.dataset, test_df, args.lookback, args.lookahead, channel_sorting)
    
    logging.info('\n')
    logging.info('Save binary Sorted dataset')
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'binary_fullset.npz'), 'bw') as trainfile:
        np.savez(trainfile, X=X_bin, y=y_bin)
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'binary_trainset.npz'), 'bw') as trainfile:
        np.savez(trainfile, X=X_train_bin, y=y_train_bin)
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'binary_testset.npz'), 'bw') as testfile:
        np.savez(testfile, X=X_test_bin, y=y_test_bin)

    logging.info('Processing Multiclass Sorted dataset')
    X_multi, y_multi, = prepare_classification_dataset(args.dataset, binned_samples_df, args.lookback, args.lookahead, channel_sorting)
    X_train_multi, y_train_multi = prepare_classification_dataset(args.dataset, train_df, args.lookback, args.lookahead, channel_sorting)
    X_test_multi, y_test_multi = prepare_classification_dataset(args.dataset, test_df, args.lookback, args.lookahead, channel_sorting)
    
    logging.info('\n')
    logging.info('Save multiclass Sorted dataset')
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'multi_fullset.npz'), 'bw') as trainfile:
        np.savez(trainfile, X=X_multi, y=y_multi)
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'multi_trainset.npz'), 'bw') as trainfile:
        np.savez(trainfile, X=X_train_multi, y=y_train_multi)
    with open(os.path.join(args.outdir,data_prefix,subdirectory,'multi_testset.npz'), 'bw') as testfile:
        np.savez(testfile, X=X_test_multi, y=y_test_multi)

    logging.info('\n')
    logging.info('Completed')