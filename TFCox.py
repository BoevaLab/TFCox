class TFCox():
    def __init__(self, seed=42,batch_norm=False,l1_ratio=1,lbda=0.0001,
                 max_it=50,learn_rate=0.001,stop_if_nan=True,stop_at_value=False, cscore_metric=False,suppress_warnings=True,verbose=0):
        
        self.max_it = max_it
        self.tnan = stop_if_nan
        self.tcscore = stop_at_value
        self.lr=learn_rate
        self.cscore=cscore_metric
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        self.l1r = l1_ratio
        self.lbda=lbda
        self.bnorm = batch_norm
        self.verbose=verbose
        if suppress_warnings == True:
            import warnings
            warnings.filterwarnings('ignore')
       
    def coxloss(self, state):
        
        def loss(y_true, y_pred):  

                return -K.mean((y_pred - K.log(tf.math.cumsum(K.exp(y_pred),reverse=True,axis=0)+0.0001))*state,axis=0)

        return loss

    def cscore_metric(self, state):
        def loss(y_true,y_pred):
            con = 0
            dis = 0
            for a in range(len(y_pred)):
                for b in range(a+1,len(y_pred)):                                       
                        if (y_pred[a]>y_pred[b])  & (y_pred[a]*state[a]!=0):
                            con+=1
                            
                        elif (y_pred[a]<y_pred[b])  & (y_pred[a]*state[a]!=0):
                            dis+=1
            return     con/(con+dis)
        return loss
 
    
    def fit(self, X,state,time):
        from tensorflow.python.framework.ops import disable_eager_execution
        disable_eager_execution()
        K.clear_session()
       
        
        
        self.time = np.array(time)  
        self.newindex = pd.DataFrame(self.time).sort_values(0).index
        self.X = (pd.DataFrame(np.array(X)).reindex(self.newindex))                      
        self.state = np.array(pd.DataFrame(np.array(state)).reindex(self.newindex))
        self.time  = np.array(pd.DataFrame(np.array(time)).reindex(self.newindex))                       
        inputsx = Input(shape=(self.X.shape[1],)) 
        state = Input(shape=(1,))
        
        if self.bnorm==True:
            out = BatchNormalization()(inputsx)
            out = Dense(1,activation='linear',
                    kernel_regularizer=l1_l2(self.lbda*self.l1r,self.lbda*(1-self.l1r)),
                   use_bias=False)(out)
        else:
            out = Dense(1,activation='linear',
                    kernel_regularizer=l1_l2(self.lbda*self.l1r,self.lbda*(1-self.l1r)),
                   use_bias=False)(inputsx)

        
        model = Model(inputs=[inputsx, state], outputs=out)
        if (self.tcscore != False) or (self.cscore==True) :
            model.compile(optimizer=Adam(self.lr) ,
                          loss=self.coxloss(state) , metrics=[self.cscore_metric(state)],
                          experimental_run_tf_function=False)
        else:
            model.compile(optimizer=Adam(self.lr) ,
                          loss=self.coxloss(state) ,
                          experimental_run_tf_function=False)
        
        self.model=model
        if self.verbose==1:
            print(self.model.summary())

        self.loss_history_ = []
        for its in range(self.max_it):
            self.temp_weights = self.model.get_weights()
           
            tr = self.model.train_on_batch([self.X, self.state],np.zeros(self.state.shape))
           
            self.loss_history_.append(tr) 
            
            if self.verbose == 1:
                if (self.tcscore != False) or (self.cscore==True) :
                    print('loss:', self.loss_history_[-1][0],' C-score: ',self.loss_history_[-1][1] )
                else:
                    print('loss:', self.loss_history_[-1] )
            
            if self.tcscore != False:
                if self.loss_history_[-1][1]>=self.tcscore:
                    print('Terminated early because concordance >=' +str(self.tcscore)+ ' as set by stop_at_value flag.')
                    break
            if (self.tcscore != False) or (self.cscore==True) :
                if (math.isnan(self.loss_history_[-1][0]) or math.isinf(self.loss_history_[-1][0])) and self.tnan:
                    self.model.set_weights(self.temp_weights)
                    print('Terminated because weights == nan or inf, reverted to last valid weight set')
                    break
            else:
                if (math.isnan(self.loss_history_[-1]) or math.isinf(self.loss_history_[-1])) and self.tnan:
                    self.model.set_weights(self.temp_weights)
                    print('Terminated because weights == nan or inf, reverted to last valid weight set')
                    break
            
        self.beta_ = self.model.get_weights()[-1]

    def predict(self,X):
        preds = self.model.predict([X,np.zeros(len(X))])

        return preds
