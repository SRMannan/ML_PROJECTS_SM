#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


(x_train , y_train) , (x_test , y_test) = keras.datasets.mnist.load_data() 
x_train = x_train/255
x_test = x_test/255


# In[31]:


x_train_flattened = x_train.reshape(len(x_train) , 28**2)
x_test_flattened = x_test.reshape(len(x_test) , 28**2)


# In[32]:


model = keras.Sequential ([
    keras.layers.Dense ( 10 , input_shape = (28**2 , ) , activation = "sigmoid")
])
model.compile(
    optimizer = "adam" , 
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)


# In[33]:


model.fit(x_train_flattened , y_train , epochs = 5)


# In[34]:


model.evaluate(x_test_flattened , y_test)


# In[38]:


plt.matshow(x_test[98])


# In[39]:


y_test_predict = model.predict(x_test_flattened)


# In[41]:


print("MODEL PREDICTS")
print(np.argmax(y_test_predict[98]))

