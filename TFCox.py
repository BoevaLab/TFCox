import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, concatenate, Lambda, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop,SGD,Nadam, Adagrad, Adadelta
from tensorflow.keras.regularizers import l1,l2,l1_l2
from tensorflow.keras.initializers import Constant ,Orthogonal, RandomNormal, VarianceScaling, Ones, Zeros
from tensorflow.keras.constraints import Constraint, UnitNorm
from keras.callbacks import Callback, TerminateOnNaN, ModelCheckpoint
import math

class TFCox():
    def __init__(self, seed=42,batch_norm=True,l1_ratio=1,lbda=0.0001,
                 max_it=50,learn_rate=0.01,stop_if_nan=True,stop_at_value=False):
        
        self.max_it = max_it
        self.tnan = stop_if_nan
        self.tcsore = stop_at_value
        self.lr=learn_rate
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        self.l1r = l1_ratio
        self.lbda=lbda
        self.bnorm = batch_norm
       
    def coxloss(self, state):
        
        def loss(y_true, y_pred):  

                return -K.mean((y_pred - K.log(tf.math.cumsum(K.exp(y_pred),reverse=True,axis=0)+0.0001))*state,axis=0)

        return loss

    def cscore(self, state):
        def loss(y_true,y_pred):
            con = 0
            dis = 0
            for a in range(len(y_pred)):
                for b in range(a+1,len(y_pred)):                   
                    if state[a]!=0:
                        if y_pred[a]>y_pred[b]:
                            con+=1
                        else:
                            dis+=1
            return     (con/(con+dis))
        return loss
 
    def fit(self, X,state,time):
        from tensorflow.python.framework.ops import disable_eager_execution
        disable_eager_execution()
        self.time = np.array(time)
        self.newindex = pd.DataFrame(self.time).sort_values(0).index
        self.X = np.array(pd.DataFrame(X).reindex(self.newindex))                      
        self.state = np.array(pd.DataFrame(state).reindex(self.newindex))
        self.time  = np.array(pd.DataFrame(time).reindex(self.newindex))                       
        inputsx = Input(shape=(self.X.shape[1],))
      
        state = Input(shape=(1,))
        if self.bnorm==True:
            out = BatchNormalization()(inputsx)
        out = Dense(1,activation='linear',
                    kernel_regularizer=l1_l2(self.lbda*self.l1r,self.lbda*(1-self.l1r)),
                   use_bias=False)(out)

        
        model = Model(inputs=[inputsx, state], outputs=out)
        model.compile(optimizer=Adam(self.lr) ,
                      loss=self.coxloss(state) , metrics=[self.cscore(state)],
                      experimental_run_tf_function=False)
       
        self.model=model
        

        self.loss_history_ = []
        for its in range(self.max_it):
            self.temp_weights = self.model.get_weights()
            self.loss_history_.append(self.model.train_on_batch([self.X, self.state],np.zeros(self.state.shape)))           
            
            
            if (self.loss_history_[-1][1]>=self.tcscore) & (self.tcscore != False):
                print('Terminated early because concordance >=' +str(self.tcscore)+ ' as set by stop_at_value flag.')
                break
        
            if (math.isnan(self.loss_history_[-1][0]) or math.isinf(self.loss_history_[-1][0])) and self.tnan:
                self.model.set_weights(self.temp_weights)
                print('Terminated because weights == nan or inf, reverted to last valid weight set')
                break
                        
        self.beta_ = self.model.get_weights()[-1]

    def predict(self,X):
        preds = self.model.predict([X,np.zeros(len(X))])

        return preds
