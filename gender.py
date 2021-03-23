import numpy as np
import tables, os, json, glob, cv2

from tqdm import tqdm
from jamo import j2hcj, h2j
from IPython.display import Image
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# Dataset Load
img_list = glob.glob("code/01_handwriting_syllable_images/1_syllable/*.png")

resized_img = []
image_id = []

for i in tqdm(range(len(img_list))): 
    if img_list[i].endswith(".png"): 
        aaa = img_list[i].split('.')[0][-8:] 
        image_id.append(aaa)
        gray = cv2.imread(img_list[i], cv2.IMREAD_GRAYSCALE)
        newsize = cv2.resize(gray, dsize=(128,128))
        resized_img.append(newsize)


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(resized_img[1])

Image(filename='code/01_handwriting_syllable_images/1_syllable/00043568.png')


# Json load
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
        syllable_id_gender.append(infos['annotations'][i]['attributes']['gender'])
        syllable_id_age.append(infos['annotations'][i]['attributes']['age'])
        syllable_id_job.append(infos['annotations'][i]['attributes']['job'])
        syllable_id_text.append(infos['annotations'][i]['text'])



syage = []
sygen = []
syjob = []
sytext = []

syage_sel = []
sygen_sel = []
syjob_sel = []
sytext_sel = []
resized_img_sel = []

for idd in image_id: 
    ind = syllable_id.index(idd)
    
    syage.append(syllable_id_age[ind]) 
    sygen.append(syllable_id_gender[ind])
    syjob.append(syllable_id_job[ind])
    sytext.append(syllable_id_text[ind])
    
# Word split
for i, k in enumerate(sytext):
    a=list(j2hcj(h2j(k))[0].split())
    if a[0] in ['ㄱ','ㄴ','ㅇ','ㅊ']:
        syage_sel.append(syllable_id_age[i])   
        sygen_sel.append(syllable_id_gender[i])
        syjob_sel.append(syllable_id_job[i])
        sytext_sel.append(k)
        
        resized_img_sel.append(resized_img[i])
assert len(sytext_sel) == len(resized_img_sel)

print(':: Number of selected image', len(sytext_sel))



v_gen = []
v_age = []

for i in range(len(sygen_sel)):
    if sygen_sel[i] == '남':
        v_gen.append(boy)
    elif sygen_sel[i] == '여':
        v_gen.append(girl)
    else:
        print("Undefined gender")

v_gen=np.asarray(v_gen)
        
print("::Number of gender list ", len(v_gen))
print("::Number of age list ", len(v_age))


norm_img=[resized_img_sel[i] / 255.0 for i in range(len(resized_img_sel))]
print("norm_img shape: ",len(norm_img))


rsh_img = [norm_img[i].reshape(128,128,1) for i in range(len(norm_img))]
rsh_img=np.asarray(rsh_img)
print("rsh_img shape: ",rsh_img.shape)



syage_sel_t = []
sygen_sel_t = []
syjob_sel_t = []
sytext_sel_t = []
resized_img_sel_t = []

for i, k in enumerate(sytext):
    a=list(j2hcj(h2j(k))[0].split())
    if a[0] in ['ㅌ']:
        syage_sel_t.append(syllable_id_age[i])   
        sygen_sel_t.append(syllable_id_gender[i])
        syjob_sel_t.append(syllable_id_job[i])
        sytext_sel_t.append(k)
        
        resized_img_sel_t.append(resized_img[i])
        
v_gen_t = []
v_age_t = []

for i in range(len(sygen_sel_t)):
    
    if sygen_sel_t[i] == '남':
        v_gen_t.append(boy)
    elif sygen_sel_t[i] == '여':
        v_gen_t.append(girl)
    else:
        print("Undefined gender")


v_gen_t=np.asarray(v_gen_t)
print("::Number of gender list ", len(v_gen_t))


norm_img_t=[resized_img_sel_t[i] / 255.0 for i in range(len(resized_img_sel_t))]
print("norm_img_t shape: ",len(norm_img_t))


rsh_img_t = [norm_img_t[i].reshape(128,128,1) for i in range(len(norm_img_t))]
rsh_img_t=np.asarray(rsh_img_t)
print("rsh_img_t shape: ",rsh_img_t.shape)

# train, test split
X_test = rsh_img_t
Y_test = v_gen_t

X_train = rsh_img
Y_train = v_gen

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)


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
model.add(Dense(2, activation = "sigmoid"))

epochs = 10 
batch_size = 250
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics=["accuracy"])

# Fit the model
history = model.fit(X_train,Y_train, batch_size=batch_size,
                              epochs = epochs, validation_data = (X_val,Y_val))

score, acc=model.evaluate(X_test, Y_test, batch_size=batch_size)
print(score, acc)


# Plot the loss and accuracy curves for training and validation 
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

