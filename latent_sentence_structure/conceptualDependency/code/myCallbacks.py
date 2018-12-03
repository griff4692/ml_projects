import keras
import numpy as np
from sklearn.metrics import roc_auc_score

class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []
        self.params['metrics'].append('val_auc')
        
    def on_train_end(self, logs={}):
        return
        
    def on_epoch_begin(self, epoch, logs={}):
        return
        
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        #y_pred = self.model.predict(self.model.validation_data[0])
        #self.aucs.append(roc_auc_score(self.model.validation_data[1], y_pred))
        
        y_pred = self.model.predict(self.validation_data[0])
        auc_epoch = roc_auc_score(self.validation_data[1], y_pred)
        logs['val_auc'] = auc_epoch
        self.aucs.append(auc_epoch)
        
        return
        
    def on_batch_begin(self, batch, logs={}):
        return
        
    def on_batch_end(self, batch, logs={}):
        return





   
