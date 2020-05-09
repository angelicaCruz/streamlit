import numpy as np 
import os
from scipy.io import loadmat
import streamlit as st
import pickle
import h5py
import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib import pyplot
import cv2
import time
import seaborn as sns


#### filenames and paths: changeable
smv_trained = os.path.join(os.getcwd(), 'svm_trained.pickle')
test_features = os.path.join(os.getcwd(), 'test_features.h5')
test_labels = os.path.join(os.getcwd(), 'test_labels.h5')
test_set = os.path.join(os.getcwd(), 'svm_testset.pickle')
original_set = os.path.join(os.getcwd(), 'data_test_color.mat')
original_label =  ['black_rot','blight','esca','healthy']
####

show_img = []
caption = []

def get_model(model_path):
	return pickle.load(open(model_path, 'rb'))

def load_images(filename):
	'''
    reads pickle files
    return: [[image, label]]
    '''
    
	images = []
	labels = []

	with open(filename, 'rb') as pickleout:
	    data = pickle.load(pickleout)

	for row in data:
	    images.append(row[0])
	    labels.append(row[1])
	    
	return images,labels

def get_h5(h5_features, h5_labels):
  h5f_data  = h5py.File(h5_features, 'r')
  h5f_label = h5py.File(h5_labels, 'r')

  features_string = h5f_data['dataset_1']
  labels_string   = h5f_label['dataset_1']

  features = np.array(features_string)
  labels   = np.array(labels_string)

  h5f_data.close()
  h5f_label.close()

  return features, labels

def classify(feature, label,img, img_name, clf):
	prediction = clf.predict(feature.reshape(1,-1))[0]
	#pred.append(prediction)
	prob = clf.predict_proba(ft.reshape(1,-1))[0]

	row = {'image_name': img_name, 'label':lbl, 'prediction':prediction, 'prob':prob}

	cap = f'pred:{original_label[prediction]} label:{original_label[lbl]} prob:{prob[prediction]:.2f}'
	show_img.append(img)
	caption.append(cap)

	return prediction, row

#### code: start

st.title('SVM DEMO')

if st.checkbox('Load SVM model', value=False):
	data_load_state = st.text('Loading model...')
	model = get_model(smv_trained)
	data_load_state.text("Done! SVM trained model available")


if st.checkbox('Load test images', value=False):
	image_load = st.text('Loading Images')
	test_X, test_y = get_h5(test_features, test_labels)
	st.text("[STATUS] features shape: {}".format(test_X.shape))
	st.text("[STATUS] labels shape: {}".format(test_y.shape))

test_images, test_lbls = load_images(test_set)

matfile = loadmat(original_set)

pred = []
results = []
count = 0

if st.checkbox('Run SVM', value=False):
	for ft,lbl,img, img_name in zip(test_X, test_y,test_images, matfile['file_names']):
		p, row = classify(ft, lbl, img, img_name, model)
		pred.append(p)

	time.sleep(2)

	#st.balloons()

	st.checkbox('Test completed', value=True)

	if st.button('Results'):
		st.write(accuracy_score(test_y, pred))	

		st.write('Confusion matrix')
		st.write(confusion_matrix(test_y, pred, normalize='true' ))


	if st.button('Show classified images'):
		st.image(show_img,caption,width=100)
		#these are the images saved during the classification!!

	


