# coding=utf-8
from __future__ import absolute_import
from tqdm import tqdm

import multiprocessing as mp
import pandas as pd
import numpy as np

import pyemblib2
import scipy
import time
import sys
import os 
from io import open

u'''
preprocessing.py

Preprocessing methods for cuto.py. 
'''

#========1=========2=========3=========4=========5=========6=========7==

def check_valid_dir(some_dir):
    if not os.path.isdir(some_dir):
        print u"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        print u"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        print u"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        print u""
        print u"DIES IST EIN UNGÜLTIGES VERZEICHNIS!!!!"
        print u""
        print u"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        print u"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        print u"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit()

#========1=========2=========3=========4=========5=========6=========7==

def check_valid_file(some_file):
    if not os.path.isfile(some_file):
        print u"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        print u"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        print u"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        print u""
        print u"DIES IST KEIN GÜLTIGER SPEICHERORT FÜR DATEIEN!!!!"
        print u""
        print u"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        print u"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        print u"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit()

#========1=========2=========3=========4=========5=========6=========7==

def loadGloveModel(gloveFile):
    print u"Loading Glove Model"
    f = open(gloveFile,u'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print u"Done.",len(model),u" words loaded!"
    return model

#========1=========2=========3=========4=========5=========6=========7==

# pass None to vocab to use use entire embedding
# RETURNS: [numpy matrix of word vectors, df of the labels]
def get_embedding_dict(emb_path, emb_format, first_n, vocab):

    print u"Preprocessing. "
    file_name_length = len(emb_path)
    extension = os.path.basename(emb_path).split(u'.')[-1]

    # Decide if it's a binary or text embedding file, and read in
    # the embedding as a dict object, where the keys are the tokens
    # (strings), and the values are the components of the corresponding 
    # vectors (floats).
    embedding = {}
    read_mode = None
    if first_n == 0 or emb_format == pyemblib2.Format.Glove:
        
        print u"No value passed for first_n or feature not supported. "
        first_n = None
    if extension == u'bin':
        read_mode = pyemblib2.Mode.Binary
        binary = True
        print u"binary reac."
    elif extension == u'txt':
        read_mode = pyemblib2.Mode.Text
        binary = False
        print u"text read."
    else:
        print u"Unsupported embedding mode. "
        exit()
    u''' 
    if emb_format == pyemblib2.Format.Glove:
        embedding = loadGloveModel(emb_path)
    '''
    
    if first_n:    
        embedding = pyemblib2.read(  emb_path, 
                                    format=emb_format,
                                    mode=read_mode,
                                    first_n=first_n,
                                    replace_errors=True,
                                    skip_parsing_errors=True,
                                    ) 
    else:
        embedding = pyemblib2.read(  emb_path, 
                                    format=emb_format,
                                    mode=read_mode,
                                    replace_errors=True,
                                    skip_parsing_errors=True,
                                    ) 
       
    return embedding

#========1=========2=========3=========4=========5=========6=========7==

# pass None to vocab to use use entire embedding
# RETURNS: [numpy matrix of word vectors, df of the labels]
def process_embedding(emb_path, emb_format, first_n, vocab):

    print u"Preprocessing. "
    file_name_length = len(emb_path)
    extension = os.path.basename(emb_path).split(u'.')[-1]

    # Decide if it's a binary or text embedding file, and read in
    # the embedding as a dict object, where the keys are the tokens
    # (strings), and the values are the components of the corresponding 
    # vectors (floats).
    embedding = {}
    read_mode = None
    if first_n == 0 or emb_format == pyemblib2.Format.Glove:
        
        print u"No value passed for first_n or feature not supported. "
        first_n = None
    if extension == u'bin':
        read_mode = pyemblib2.Mode.Binary
        binary = True
        print u"binary reac."
    elif extension == u'txt':
        read_mode = pyemblib2.Mode.Text
        binary = False
        print u"text read."
    else:
        print u"Unsupported embedding mode. "
        exit()
    u''' 
    if emb_format == pyemblib2.Format.Glove:
        embedding = loadGloveModel(emb_path)
    '''
    
    if first_n:    
        embedding = pyemblib2.read(  emb_path, 
                                    format=emb_format,
                                    mode=read_mode,
                                    first_n=first_n,
                                    replace_errors=True,
                                    skip_parsing_errors=True,
                                    ) 
    else:
        embedding = pyemblib2.read(  emb_path, 
                                    format=emb_format,
                                    mode=read_mode,
                                    replace_errors=True,
                                    skip_parsing_errors=True,
                                    ) 
       
     
    
    # take a subset of the vocab
    new_embedding = {}
    if (vocab != None):
        for word in vocab:
            if word in embedding:
                vector = embedding[word]
                new_embedding.update({word:vector})
        embedding = new_embedding

    # convert embedding to pandas dataframe
    # "words_with_friends" is the column label for the vectors
    # this df has shape [num_inputs,2] since the vectors are all in 1
    # column as length d lists 
    emb_array = np.array(embedding.items())
    print u"numpy"
    sys.stdout.flush() 

    label_array = np.array([ row[0] for row in emb_array.tolist() ])
    #print(label_array[0:10])
    sys.stdout.flush() 
    
    vectors_matrix = np.array([ row[1:] for row in emb_array.tolist() ])
    vectors_matrix = np.array([ row[0] for row in vectors_matrix ])
    #print(vectors_matrix[0:10])
    sys.stdout.flush() 

    u'''
    emb_df = pd.Series(embedding, name="words_with_friends")
    # print(emb_df.head(10))

    # reset the index of the dataframe
    emb_df = emb_df.reset_index()
    # print(emb_df.head(10))

    # matrix of just the vectors
    emb_matrix = emb_df.words_with_friends.values.tolist()
    # print(emb_matrix[0:10])

    # dataframe of just the vectors
    vectors_df = pd.DataFrame(emb_matrix,index=emb_df.index)
    # print(vectors_df.head(10))

    # numpy matrix of just the vectors
    vectors_matrix = vectors_df.as_matrix()
    # print(vectors_matrix[0:10])
    '''

    return vectors_matrix, label_array

#========1=========2=========3=========4=========5=========6=========7==

# pass None to vocab to use use entire embedding
# DOES: Saves the first n words in a new embedding file
def subset_embedding(emb_path, first_n, vocab):

    print u"Preprocessing. "
    file_name_length = len(emb_path)
    last_char = emb_path[file_name_length - 1]

    # Decide if it's a binary or text embedding file, and read in
    # the embedding as a dict object, where the keys are the tokens
    # (strings), and the values are the components of the corresponding 
    # vectors (floats).
    embedding = {}
    if (last_char == u'n'):
        embedding = pyemblib2.read(emb_path, 
                                  mode=pyemblib2.Mode.Binary,
                                  first_n=first_n) 
    elif (last_char == u't'):
        embedding = pyemblib2.read(emb_path, 
                                  mode=pyemblib2.Mode.Text, 
                                  first_n=first_n)
    else:
        print u"Unsupported embedding format. "
        exit()

    # make sure it has a valid file extension
    extension = emb_path[file_name_length - 4:file_name_length]
    if extension != u".txt" and extension != u".bin":
        print u"Invalid file path. "
        exit()
   
    # get the emb_path without the file extension 
    path_no_ext = emb_path[0:file_name_length - 4]
    new_path = path_no_ext + u"_SUBSET.txt"

    # write to text embedding file
    pyemblib2.write(embedding, 
                   new_path, 
                   mode=pyemblib2.Mode.Text)
    
    return 
