from keras.callbacks import EarlyStopping, Callback


class CustomPrintCallback(Callback):
    """Custom Callback printing useful metrics-related information for each epoch."""
    
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
    
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}/{self.epochs} - loss: {logs['loss']:.4f} - avg_f1: {logs['avg_f1']:.4f} - val_loss: {logs['val_loss']:.4f} - val_avg_f1: {logs['val_avg_f1']:.4f}\n")


class CustomBestEarlyStopping(EarlyStopping):
    """Custom EarlyStopping to restore best model weights on train end."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"\nEpoch {self.stopped_epoch + 1}: early stopping.")
        elif self.restore_best_weights:
            if self.verbose > 0:
                print("Restoring best model weights.")
            self.model.set_weights(self.best_weights)