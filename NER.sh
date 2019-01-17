#!/bin/bash
#source activate theano
#srun -J glampleConvertToTxt --mem 2000 -c 2 -w locomotion python binConverter.py

# script to run with pretrained embedding (dist embedding)
srun -J glampleTrain --mem 10000 -c 4 -w adamantium python /u/user/tagger-v/train.py --train=eng.train --dev=eng.testa --test=eng.testb --pre_emb=../steve_embedding.txt

# script to run with pretrained embedding (random initializations)
#srun -J glampleTrain --mem 30000 -c 20 -w osmium python /u/user/tagger-v/train.py --train=eng.train --dev=eng.testa --test=eng.testb 

# srun -J lampRun --mem 3500 -c 4 -w feldspar python /u/user/tagger-v/tagger.py --model models/tag_scheme=iobes,lower=False,zeros=False,char_dim=25,char_lstm_dim=25,char_bidirect=True,word_dim=100,word_lstm_dim=100,word_bidirect=True,pre_emb=gigatext1.txt,all_emb=False,cap_dim=0,crf=True,dropout=0.5,lr_method=sgd-lr_.005 --input eng.testb --output nerPreds.txt

#srun -J tagger --mem 25000 -c 20 -w osmium python /u/user/tagger-v/tagger.py --model models/tag_scheme=iobes,lower=False,zeros=False,char_dim=25,char_lstm_dim=25,char_bidirect=True,word_dim=100,word_lstm_dim=100,word_bidirect=True,pre_emb=steve_embedding.txt,all_emb=False,cap_dim=0,crf=True,dropout=0.5,lr_method=sgd-lr_.005 --input eng.testb --output nerPreds_dist.txt

