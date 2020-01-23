# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 10:22:27 2018

@author: Yang
"""
'''
Dataset:
    - https://ti.arc.nasa.gov/tech/dash/pcoe/prognostic-data-repository/#turbofan
'''

import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import matplotlib as mpl

import os
from sklearn import preprocessing

from sklearn.metrics import mean_squared_error
from math import sqrt, expm1

import csv
import time


# Setting seed for reproducibility
np.random.seed(1234)  
PYTHONHASHSEED = 0

numDataset = str(1)
batchsize = 10 #depends on min(eng_cycle)

##################################
# Data Ingestion
##################################

# read training data - It is the aircraft engine run-to-failure data.
datasetPath = "./CMAPSSData/"
comp_dataPath = "/mnt/storage/scratch/yz8904/comp4_LSTM_test/compression_results/comp_res_merge/"

for loop_bitdepth in range(4, 18, 2): # loop bitdepth: 4,6,8,10,12,14,16
	train_df = pd.read_csv(comp_dataPath+'rec_train_FD00%s_%dbitdepth.txt' % (numDataset, loop_bitdepth) , sep=" ", header=None)
#	train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
	train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
						 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
						 's15', 's16', 's17', 's18', 's19', 's20', 's21']

	train_df = train_df.sort_values(['id','cycle'])

	# read test data - It is the aircraft engine operating data without failure events recorded.
	test_df = pd.read_csv(datasetPath+'test_FD00%s.txt' % numDataset, sep=" ", header=None)
	test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
	test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
						 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
						 's15', 's16', 's17', 's18', 's19', 's20', 's21']

	# read ground truth data - It contains the information of true remaining cycles for each engine in the testing data.
	truth_df = pd.read_csv(datasetPath+'RUL_FD00%s.txt' % numDataset, sep=" ", header=None)
	truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)

	##################################
	# Data Preprocessing
	##################################

	#######
	# TRAIN
	#######
	# Data Labeling - generate column RUL(Remaining Usefull Life or Time to Failure)
	rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
	rul.columns = ['id', 'max']
	train_df = train_df.merge(rul, on=['id'], how='left')
	train_df['RUL'] = train_df['max'] - train_df['cycle']
	train_df.drop('max', axis=1, inplace=True)

	w1 = 30
	w0 = 15
	train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0 )
	train_df['label2'] = train_df['label1']
	train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2

	# MinMax normalization (from 0 to 1)
	train_df['cycle_norm'] = train_df['cycle']
	cols_normalize = train_df.columns.difference(['id','cycle','RUL','label1','label2'])
	min_max_scaler = preprocessing.MinMaxScaler()
	norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), 
								 columns=cols_normalize, 
								 index=train_df.index)
	join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
	train_df = join_df.reindex(columns = train_df.columns)

	#train_df.to_csv('../../Dataset/PredictiveManteinanceEngineTraining.csv', encoding='utf-8',index = None)

	######
	# TEST
	######
	# MinMax normalization (from 0 to 1)
	test_df['cycle_norm'] = test_df['cycle']
	norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]), 
								columns=cols_normalize, 
								index=test_df.index)
	test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
	test_df = test_join_df.reindex(columns = test_df.columns)
	test_df = test_df.reset_index(drop=True)
	print(test_df.head())

	rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
	rul.columns = ['id', 'max']
	truth_df.columns = ['more']
	truth_df['id'] = truth_df.index + 1
	truth_df['max'] = rul['max'] + truth_df['more']
	truth_df.drop('more', axis=1, inplace=True)

	# generate RUL for test data
	test_df = test_df.merge(truth_df, on=['id'], how='left')
	test_df['RUL'] = test_df['max'] - test_df['cycle']
	test_df.drop('max', axis=1, inplace=True)

	# generate label columns w0 and w1 for test data
	test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0 )
	test_df['label2'] = test_df['label1']
	test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2

	def plot_1(test_df,diff_pred_true,loopsequence_length,save_path):
	#    plot length of each engine in dataset against the pred diff to true_test
		
		eng_cycle = []
		for id in test_df['id'].unique():
			eng_cycle = eng_cycle + [len(test_df[test_df['id']==id])]
		idx = np.asarray([i for i in range(len(eng_cycle))])
		df = pd.DataFrame({'Engine_Cycle':eng_cycle,'Diff_Pred_True':diff_pred_true[:,0]})
		
		#csfont = {'fontname':'Arial'}
		
		fig = plt.figure(figsize=(45,10)) # Create matplotlib figure
		ax = fig.add_subplot(111) # Create matplotlib axes
		ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
		
		width = 0.4
		
		df.Engine_Cycle.plot(kind='bar', color='red', ax=ax, width=width, position=1,label="Engine Cycles")
		df.Diff_Pred_True.plot(kind='bar', color='blue', ax=ax2, width=width, position=0, label="Diff Pred/True")
		
		lines, labels = ax.get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		ax.legend(lines + lines2, labels + labels2)
		
		ax.set_ylabel('Engine cycles in testset')
		ax.set_xlabel('Engine ID')
		ax2.set_ylabel('Diff between pred vs true')
		
		ax.set_ylim([0, 350])
		ax2.set_ylim([0, 350])
		
		#plt.xticks( np.arange(len(test_df)))
		plt.xticks(idx, idx+1, rotation='vertical')
		
		plt.title('Engine cycle vs Diff Pred/True')
		
		mpl.rcParams.update({'font.size': 15})
		mpl.rc('font',family='Consolas')
		
		plt.show()
		save_fig_str = save_path + "seqLengthTestset_vs_diffPredTrue_%s.pdf" % str(loopsequence_length)
		fig.savefig(save_fig_str, bbox_inches='tight')
		


	def plot_(test_df,diff_pred_true,loopsequence_length,save_path):
	#   plot length of each engine in testset
		if loopsequence_length == batchsize:
			eng_cycle = []
			for id in test_df['id'].unique():
				eng_cycle = eng_cycle + [len(test_df[test_df['id']==id])]
			fig_engCycle = plt.figure(figsize=(20,10))
			plt.bar(range(len(eng_cycle)), eng_cycle, color='rgb')  
			plt.title('Sequence length in testset for each engine')
			plt.ylabel('Length')
			plt.xlabel('Engine ID')
			
			axes = plt.gca()
			axes.set_ylim([0, 80])
			
			plt.show()
			fig_engCycle.savefig(save_path + 'seqLength_testset.png')

	#    plot diff between the pred_test and true_test
		diff_list = []
	#    diff_pred_true = np.absolute(diff_pred_true)
		for loopDiff in range(len(diff_pred_true)):
			diff_list = diff_list + diff_pred_true[loopDiff].tolist()
		
		fig_diff = plt.figure(figsize=(20,10))
		plt.bar(range(len(diff_pred_true)), diff_list, color='rgb')  
		plt.title('Difference between Predict RUL and True RUL (seqLen = %s)' %  str(loopsequence_length))
		plt.ylabel('Diff RUL')
		plt.xlabel('samples')
		plt.show()
		save_fig_str = save_path + "diff_pred_true_%s.png" % str(loopsequence_length)
		fig_diff.savefig(save_fig_str)





	def s_scoring(estRUL, trueRUL):
		h = estRUL - trueRUL
		s_out = np.zeros((len(h),1), float)
		for loop_h in range(len(h)):
			if h[loop_h]<0:
				s = expm1(-h[loop_h]/13)
			else:
				s = expm1(h[loop_h]/10)
			s_out[loop_h] = s
		s_out_ = sum(s_out)
		return (s_out, s_out_)


	def store_true_(old, mask_, new, *therest):
		point_ = 0
		old_temp = old.copy()
		if 'better_map' in globals():
			for loopData in range(len(better_map)):
				if better_map[loopData] ==0:
					old_temp[loopData] = new[loopData]
		else:
			for loopData in range(len(mask_)):
				if mask_[loopData] ==1:
					old_temp[loopData] = new[point_]
					point_ += 1
		fusion_out = old_temp
		return fusion_out


	exeTime = []
	for loopBatchSize in range(100, 1300, 200): # batch size from 100-500, step=100
		start_time = time.time() #get start time
		
		save_path = './Output/res11_1LSTMlayer/FD00%s/FD00%s_%dbitdepth_batch%s_ensemble/' % (numDataset,numDataset,loop_bitdepth,str(loopBatchSize))

		if not os.path.exists(save_path):
			os.makedirs(save_path)
		# define path to save model
		model_path = save_path + 'regression_model.h5'

		#test_df.to_csv('../../Dataset/PredictiveManteinanceEngineValidation.csv', encoding='utf-8',index = None)
		cat_rmse = np.array([])
		cat_rmse_test_fusion = np.array([])
		cat_s_single_score = np.array([])
		cat_str = np.array([])
	#    for loopsequence_length in range(30,130,10):
		for loopsequence_length in range(batchsize,130,20):
		# pick a large window size of 50 cycles
		#sequence_length = 80
			sequence_length = loopsequence_length
			# function to reshape features into (samples, time steps, features) 
			def gen_sequence(id_df, seq_length, seq_cols):
				data_matrix = id_df[seq_cols].values
				num_elements = data_matrix.shape[0]
				# Iterate over two lists in parallel.
				for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
					yield data_matrix[start:stop, :]
					
			# pick the feature columns 
			sensor_cols = ['s' + str(i) for i in range(1,22)]
			sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
			sequence_cols.extend(sensor_cols)
			
			# TODO for debug 
			val=list(gen_sequence(train_df[train_df['id']==1], sequence_length, sequence_cols))
			print(len(val))
			
			# generator for the sequences
			# transform each id of the train dataset in a sequence
			seq_gen = (list(gen_sequence(train_df[train_df['id']==id], sequence_length, sequence_cols)) 
					   for id in train_df['id'].unique())
			
			seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
			print(seq_array.shape)
			
			# function to generate labels
			def gen_labels(id_df, seq_length, label):
				
				data_matrix = id_df[label].values
				num_elements = data_matrix.shape[0]
				
				return data_matrix[seq_length:num_elements, :]
			
			# generate labels
			label_gen = [gen_labels(train_df[train_df['id']==id], sequence_length, ['RUL']) 
						 for id in train_df['id'].unique()]
			
			label_array = np.concatenate(label_gen).astype(np.float32)
			label_array.shape
			
			##################################
			# Modeling
			##################################
			
			def r2_keras(y_true, y_pred):
				"""Coefficient of Determination 
				"""
				SS_res =  K.sum(K.square( y_true - y_pred ))
				SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
				return ( 1 - SS_res/(SS_tot + K.epsilon()) )
			
			nb_features = seq_array.shape[2]
			nb_out = label_array.shape[1]
			
			model = Sequential()
			model.add(LSTM(
					  input_shape=(sequence_length, nb_features),
					  units=100,
					  return_sequences=False))
			model.add(Dropout(0.2))
			model.add(Dense(units=nb_out))
			model.add(Activation("linear"))
			model.compile(loss='mean_squared_error', optimizer='rmsprop',metrics=['mae',r2_keras])

			print(model.summary())

			history = model.fit(seq_array, label_array, epochs=400, batch_size=loopBatchSize, validation_split=0.05, verbose=2,
					  callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
								   keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
					  )
			
			print(history.history.keys())
			
			fig_acc = plt.figure(figsize=(10, 10))
			plt.plot(history.history['r2_keras'])
			plt.plot(history.history['val_r2_keras'])
			plt.title('model r^2')
			plt.ylabel('R^2')
			plt.xlabel('epoch')
			plt.legend(['train', 'test'], loc='upper left')
			plt.show()
			fig_acc.savefig(save_path + "model_r2.png")
			
			# summarize history for MAE
			fig_acc = plt.figure(figsize=(10, 10))
			plt.plot(history.history['mean_absolute_error'])
			plt.plot(history.history['val_mean_absolute_error'])
			plt.title('model MAE')
			plt.ylabel('MAE')
			plt.xlabel('epoch')
			plt.legend(['train', 'test'], loc='upper left')
			plt.show()
			fig_acc.savefig(save_path + "model_mae.png")
			
			# summarize history for Loss
			fig_acc = plt.figure(figsize=(10, 10))
			plt.plot(history.history['loss'])
			plt.plot(history.history['val_loss'])
			plt.title('model loss')
			plt.ylabel('loss')
			plt.xlabel('epoch')
			plt.legend(['train', 'test'], loc='upper left')
			plt.show()
			fig_acc.savefig(save_path + "model_regression_loss_%s.png"  % str(loopsequence_length))
			
			# training metrics
			scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=100)
			print('\nMAE: {}'.format(scores[1]))
			print('\nR^2: {}'.format(scores[2]))
			
			y_pred = model.predict(seq_array,verbose=1, batch_size=100)
			y_true = label_array
			
			test_set = pd.DataFrame(y_pred)
			test_set.to_csv(save_path + 'submit_train.csv', index = None)
			
			##################################
			# EVALUATE ON TEST DATA
			##################################
			
			seq_array_test_last = [test_df[test_df['id']==id][sequence_cols].values[-sequence_length:] 
								   for id in test_df['id'].unique() if len(test_df[test_df['id']==id]) >= sequence_length]
			
			seq_array_test_all = [test_df[test_df['id']==id][sequence_cols].values[-sequence_length:] 
								   for id in test_df['id'].unique() if len(test_df[test_df['id']==id]) >= 0]
			
			seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
			print("seq_array_test_last")
			
			print(seq_array_test_last.shape)
			
			y_mask = [len(test_df[test_df['id']==id]) >= sequence_length for id in test_df['id'].unique()]
			label_array_test_last = test_df.groupby('id')['RUL'].nth(-1)[y_mask].values
			label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)
			print(label_array_test_last.shape)
			print("label_array_test_last")
			print(label_array_test_last)
			
			# pick all the labels for final result evaluation
			y_no_mask = [len(test_df[test_df['id']==id]) >= 0 for id in test_df['id'].unique()]
			label_array_test_last_whole = test_df.groupby('id')['RUL'].nth(-1)[y_no_mask].values
			label_array_test_last_whole = label_array_test_last_whole.reshape(label_array_test_last_whole.shape[0],1).astype(np.float32)
			y_true_test_whole = label_array_test_last_whole
			
			if os.path.isfile(model_path):
				estimator = load_model(model_path,custom_objects={'r2_keras': r2_keras})
			
				# test metrics
				scores_test = estimator.evaluate(seq_array_test_last, label_array_test_last, verbose=2)
				print('\nMAE: {}'.format(scores_test[1]))
				print('\nR^2: {}'.format(scores_test[2]))
			
				y_pred_test = estimator.predict(seq_array_test_last)
				y_pred_test = np.array(y_pred_test).round() #due to RUL is int, so round to int
				y_true_test = label_array_test_last
				
				if not 'y_fusion' in globals() or ('y_fusion' in globals() and loopsequence_length == batchsize):
					y_fusion = np.zeros((len(y_mask),1), dtype=float)
				
				if loopsequence_length == batchsize:
					y_pred_test_fusion = store_true_(y_fusion, y_mask, y_pred_test)
				else:
					y_fusion_temp = y_pred_test_fusion.copy()
					y_pred_test_fusion_now = store_true_(y_fusion_temp, y_mask, y_pred_test)
					diff_pred_true_now = abs(y_pred_test_fusion_now - y_true_test_whole)
					pos_neg = diff_pred_true - diff_pred_true_now
					
					better_map = [pos_neg[loopPos_neg] > 0 for loopPos_neg in range(len(pos_neg))]
					y_pred_test_fusion_best = store_true_(y_pred_test_fusion_now, y_mask, y_pred_test_fusion, better_map)
					y_pred_test_fusion = y_pred_test_fusion_best.copy()
					
				if 'better_map' in globals():
						del better_map
				
				rmse_test = sqrt(mean_squared_error(y_pred_test, y_true_test))
				print('\nRMSE: ', rmse_test)
						
				test_set = pd.DataFrame(y_pred_test)
				test_set.to_csv(save_path+'submit_test.csv', index = None)
				
			
				# Plot in blue color the predicted data and in green color the
				# actual data to verify visually the accuracy of the model.
				fig_verify = plt.figure(figsize=(20, 10))
				plt.plot(y_pred_test, color="blue")
				plt.plot(y_true_test, color="green")
				plt.title('prediction')
				plt.ylabel('value')
				plt.xlabel('row')
				plt.legend(['predicted', 'actual data'], loc='upper left')
				plt.show()
				save_fig_str = save_path + "model_regression_verify_%s.png" % str(loopsequence_length)
				fig_verify.savefig(save_fig_str)
				
				if len(y_pred_test_fusion) == len(y_true_test_whole):
					diff_pred_true = abs(y_pred_test_fusion - y_true_test_whole)
					plot_(test_df,diff_pred_true,loopsequence_length,save_path) #plot length of each test engen and diff between pred and true
					plot_1(test_df,diff_pred_true,loopsequence_length,save_path)
					
					rmse_test_fusion = sqrt(mean_squared_error(y_pred_test_fusion, y_true_test_whole))
					
					save_rmse_test_fusion = pd.DataFrame([rmse_test_fusion])
					save_rmse_test_fusion.to_csv(save_path+'rmse_test_fusion_%s.csv' % str(loopsequence_length), index = None)
					save_y_pred_test_fusion = pd.DataFrame(y_pred_test_fusion)
					save_y_pred_test_fusion.to_csv(save_path+'y_pred_test_fusion_%s.csv' % str(loopsequence_length), index = None)
					# calculate s scoring metric Babu et al. 2016 Eq(1)
					s_fusion, s_single_score = s_scoring(y_pred_test_fusion, y_true_test_whole)
					test_s_fusion = pd.DataFrame(s_fusion)
					test_s_fusion.to_csv(save_path+'s_fusion_%s.csv' % str(loopsequence_length), index = None)
					test_s_single_score = pd.DataFrame(s_single_score)
					test_s_single_score.to_csv(save_path+'s_single_score_%s.csv' % str(loopsequence_length), index = None)
					
					
					# Plot in blue color the predicted data and in green color the
						# actual data to verify visually the accuracy of the model.
					fig_fusionRes = plt.figure(figsize=(20, 10))
					plt.plot(y_pred_test_fusion, color="blue")
					plt.plot(y_true_test_whole, color="green")
					plt.title('Fusion prediction results')
					plt.ylabel('value')
					plt.xlabel('row')
					plt.legend(['predicted', 'actual data'], loc='upper left')
					plt.show()
					save_fig_str = save_path + "model_regression_fusionRes_%s.png" % str(loopsequence_length)
					fig_fusionRes.savefig(save_fig_str)
				else:
					print('Length of y_ture_test = ' + str(len(y_true_test_whole)))
					print('Length of y_pred_test_fusion = ' + str(len(y_pred_test_fusion)))
		
			cat_rmse = np.append(cat_rmse, rmse_test)
			cat_rmse_test_fusion = np.append(cat_rmse_test_fusion, rmse_test_fusion)
			cat_s_single_score = np.append(cat_s_single_score, s_single_score[0])
			cat_str = np.append(cat_str, 'seq length %s' % str(loopsequence_length))
			
	#        clear session to avoid tensorflow memory leak
			if K.backend() == 'tensorflow':
				K.clear_session()
			
		fig_rmse = plt.figure(figsize=(10,10))
		plt.plot(cat_rmse, color="blue")
		plt.title('RMSE result, seq_length ranging [%s 120]' % str(batchsize))
		plt.ylabel('RMSE')
		plt.xlabel('samples')
		plt.show()
		fig_rmse.savefig(save_path+"RMSE_seqLength_%s_120.png" % str(batchsize))
		
		csv_fileHeader = ['seq_length','rmse', 's_score_fusion','rmse_fusion']
		csv_rows = zip(cat_str.tolist(), cat_rmse, cat_s_single_score,cat_rmse_test_fusion)
		with open(save_path + 'fusionResults.csv', "w") as f:
			writer = csv.writer(f)
			writer.writerow(csv_fileHeader)
			for row in csv_rows:
				writer.writerow(row)
		
		print('Final results list below:')
		print('RMSE for all = ' + str(rmse_test_fusion))
		print('S score for all = ' + str(s_single_score[0]))
		
		exeTime = exeTime + [time.time()-start_time]
print('Finished')