#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tables
import os
import json 
import glob
import cv2
from tqdm import tqdm
from jamo import j2hcj, h2j
from IPython.display import Image
import matplotlib.pyplot as plt


# In[2]:


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# In[4]:


img_list = glob.glob("code/01_handwriting_syllable_images/1_syllable/*.png") 

# resize image

resized_img = []
image_id = []

for i in tqdm(range(len(img_list))): 
    if img_list[i].endswith(".png"): 
        aaa = img_list[i].split('.')[0][-8:] 
        image_id.append(aaa)
        gray = cv2.imread(img_list[i], cv2.IMREAD_GRAYSCALE)
        newsize = cv2.resize(gray, dsize=(128,128)) 
        resized_img.append(newsize)


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(resized_img[1])


# In[7]:


info_file = open('handwriting_data_info1.json', encoding='UTF8').read() 

infos = json.loads(info_file)
data_N = len(infos['annotations'])

syllable_id = [] 
syllable_id_gender = []
syllable_id_age = []
syllable_id_job = []
syllable_id_text = []

for i in range(data_N):
    
    if infos['annotations'][i]['attributes']['type'] == '글자(음절)':
        syllable_id.append(infos['annotations'][i]['id'])
        syllable_id_age.append(infos['annotations'][i]['attributes']['age'])
        syllable_id_text.append(infos['annotations'][i]['text'])


# In[9]:


syage = []
sytext = []

syage_sel = []
sytext_sel = []
resized_img_sel = []

for idd in image_id:
    
    ind = syllable_id.index(idd)
    
    syage.append(syllable_id_age[ind])   
    sytext.append(syllable_id_text[ind])
    
for i, k in enumerate(sytext):
    a=list(j2hcj(h2j(k))[0].split())
    if a[0] in ['ㄱ','ㄴ','ㄹ','ㅈ','ㅉ']:
        syage_sel.append(syllable_id_age[i])   
        sytext_sel.append(k)
        
        resized_img_sel.append(resized_img[i])


syage_sel = syage_sel[:7100]
sytext_sel = sytext_sel[:7100]
resized_img_sel= resized_img_sel[:7100]

print(':: Number of selected image', len(sytext_sel))

boy = [1, 0] 
girl = [0, 1]
twe_erl = [1,0,0,0] 
twe_lat = [0,1,0,0] 
thr_erl = [0,0,1,0] 
thr_lat = [0,0,0,1] 


# In[11]:


v_age = []

for i in range(len(syage_sel)):
    if 22<= int(syage_sel[i]) <= 25 :
        v_age.append(twe_erl)
    elif 26<= int(syage_sel[i]) <= 29:
        v_age.append(twe_lat)
    elif 30<=int(syage_sel[i]) <= 34:
        v_age.append(thr_erl)
    elif 36< int(syage_sel[i]):
         v_age.append(thr_lat)
    else:
        print("Undefined age") 

v_age=np.asarray(v_age)

print("::Number of age list ", len(v_age))


# In[12]:



norm_img=[resized_img_sel[i] / 255.0 for i in range(len(resized_img_sel))]
print("norm_img shape: ",len(norm_img))


# In[13]:



rsh_img = [norm_img[i].reshape(128,128,1) for i in range(len(norm_img))]
rsh_img=np.asarray(rsh_img)
print("rsh_img shape: ",rsh_img.shape)


# In[14]:


X_train, X_test, Y_train, Y_test = train_test_split(rsh_img, v_age, test_size = 0.1, random_state=2)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
print("X_train shape",X_train.shape)
print("X_val shape",X_val.shape)
print("X_test shape",X_test.shape)
print("Y_train shape",Y_train.shape)
print("Y_val shape",Y_val.shape)
print("Y_test shape",Y_test.shape)

# train 6714 10738
# val 697 1243
# test 849 1306


# In[8]:


def _permutation(set):
    permid = np.random.permutation(len(set[0]))
    for i in range(len(set)):
        set[i] = set[i][permid]

    return set


# In[15]:


#Model
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (128,128,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(4, activation = "softmax"))


# In[16]:


epochs = 10  # for better result increase the epochs
batch_size = 250


# In[2]:


# Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)


# In[18]:



model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[19]:


history = model.fit(X_train,Y_train, batch_size=batch_size,
                              epochs = epochs, validation_data = (X_val,Y_val))


# In[20]:


score, acc=model.evaluate(X_test, Y_test, batch_size=batch_size)
print(score, acc)


# In[22]:



plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

