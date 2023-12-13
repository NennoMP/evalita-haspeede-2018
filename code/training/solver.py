import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator
from keras.losses import BinaryCrossentropy
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold                 
        
from training.callbacks import CustomPrintCallback, CustomBestEarlyStopping

class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training a classification/regression model.
    It performs gradient descent using the given learning rate.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    After training (i.e. when train() method returns), the best performing weights are loaded back into
    the model, and the best achieved stats are stored in best_model_stats. History records for train loss,
    val loss, train F1-score, and val F1-score are also memorized.
    """
    
    def __init__(self, model, train_data, train_labels, val_data, val_labels, target='val_avg_f1', **kwargs):
        """
        Construct a new Solver instance.

        Args:
        - model: a model object

        - train_data: training data
        - train_labels: training gtc labels
        
        - val_data: validation data
        - val_labels: validation gtc labels

        - target: target metric for gradient descent
        """
        
        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels
        self._reset()
        
        self.target = target
    
    def _reset(self):
        """
        Reset/Setup some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.best_model_stats = None
        self.best_params = None

        self.train_loss_history = []
        self.val_loss_history = []
        
        # Record F1 history
        self.train_metric_history = []
        self.val_metric_history = []
        
    def plot_history(self, out_path: str):
        epochs = list(range(1, len(self.train_loss_history) + 1))

        plt.figure(figsize=(12, 5))

        # Plotting BCE losses
        ax1 = plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot = Losses
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.plot(epochs, self.train_loss_history, label='Training', marker='o')
        ax1.plot(epochs, self.val_loss_history, label='Validation', marker='o')
        ax1.set_title('BCE losses')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Plotting average F1-scores
        ax2 = plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot = F1 Scores
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.plot(epochs, self.train_metric_history, label='Trainining', marker='o')
        ax2.plot(epochs, self.val_metric_history, label='Validation', marker='o')
        ax2.set_title('Average F1-scores')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('average F1')
        ax2.legend()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.show()

    def train(self, epochs=50, batch_size=32, patience=None):
        """
        Run optimization to train the model.
        """
        
        if self.target == 'val_avg_f1':
            mode = 'max'
        else:
            raise ValueError(f"Unsupported TARGET value: {self.target}. Try 'val_avg_f1!'")
        
        
        callbacks = [CustomPrintCallback(epochs=epochs)]
        
        if patience:
            early_stopping = CustomBestEarlyStopping(monitor=self.target, patience=patience, mode=mode, verbose=1, restore_best_weights=True)
            callbacks.append(early_stopping)

        history = self.model.fit(self.train_data, self.train_labels,
                                 validation_data=(self.val_data, self.val_labels),
                                 epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
        
        # Record history loss
        self.train_loss_history = history.history['loss']
        self.val_loss_history = history.history['val_loss']
        
        # Record history average F1
        self.train_metric_history = history.history['avg_f1']
        self.val_metric_history = history.history['val_avg_f1']
        
        if self.target == 'val_avg_f1':
            best_target_idx = np.argmax(self.val_metric_history)
        else:
            raise ValueError(f'Unsupported TARGET value: {self.target}. Try <val_avg_f1>!')
        self.best_model_stats = {'val_loss': self.val_loss_history[best_target_idx],
                                 'train_loss': self.train_loss_history[best_target_idx],
                                 'val_avg_f1': self.val_metric_history[best_target_idx],
                                 'avg_f1': self.train_metric_history[best_target_idx]}
        
        print(f"Best validation macro F1-score: {self.best_model_stats['val_avg_f1']}")
        
    def train_with_kfold(self, model_fn, hparams, dev_data, epochs=50, batch_size=32, patience=None, n_splits=5):
        self.models = []  # store each fold's model
        self.all_histories = []
        self.all_metrics = []

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=128)
        fold = 0
        
        for train_idx, val_idx in kf.split(dev_data['text'], dev_data['label']):
            fold += 1
            print(f'Training on fold {fold}/{n_splits}')
            
            callbacks = [CustomPrintCallback(epochs=epochs)]
            if patience:
                early_stopping = CustomBestEarlyStopping(monitor='val_avg_f1', patience=patience, mode='max', verbose=1, restore_best_weights=True)
                callbacks.append(early_stopping)

            # Split data into training and validation
            input_train_kfold = {'text': dev_data['text'][train_idx], 'PoS': dev_data['PoS'][train_idx], 'extra': dev_data['extra'][train_idx]}
            y_train_kfold = dev_data['label'][train_idx]

            input_val_kfold = {'text': dev_data['text'][val_idx], 'PoS': dev_data['PoS'][val_idx], 'extra': dev_data['extra'][val_idx]}
            y_val_kfold = dev_data['label'][val_idx]

            # Get a new model instance and compile
            model = model_fn(hparams)

            # Train and store the history
            history = model.fit(input_train_kfold, y_train_kfold,
                                validation_data=(input_val_kfold, y_val_kfold), 
                                epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)

            self.all_histories.append(history)

            # Evaluate and store score
            scores = model.evaluate(input_val_kfold, y_val_kfold)
            self.all_metrics.append(scores)

            # Save the model for ensemble prediction later
            self.models.append(model)

        # Average scores after K-fold CV
        average_scores = np.mean(self.all_metrics, axis=0)
        print(f'Average scores across {n_splits} folds: {average_scores}')
        
        return self.models
        
    def ensemble_predict(self, data):
        predictions = []
        for model in self.models:
            # Predict and threshold to get class labels (0 or 1)
            model_prediction = np.where(model.predict(data) > 0.5, 1, 0)
            predictions.append(model_prediction)

        # Stack predictions for easier counting
        stacked_predictions = np.hstack(predictions)

        # Count 0s and 1s for each sample and decide based on majority
        ensemble_class_labels = []
        for sample_preds in stacked_predictions:
            counts = np.bincount(sample_preds)
            ensemble_class_labels.append(np.argmax(counts))

        return np.array(ensemble_class_labels)
    
    def meta_learner_predict(self, data, meta_learner):
        # Create empty arrays to store predictions from base models
        predictions = []

        # Generate predictions from your base models for the input test data
        for model in self.models:
            val_predictions = model.predict(self.val_data)
            predictions.append(val_predictions)

        # Stack the predictions horizontally
        stacked_predictions = np.column_stack(predictions)

        # Train a meta-learner (e.g., logistic regression) on base model predictions
        meta_learner.fit(stacked_predictions, self.val_labels)

        # Make predictions on your test data using the trained meta-learner
        test_predictions = []
        for model in self.models:
            test_predictions.append(model.predict(data))

        stacked_test_predictions = np.column_stack(test_predictions)
        final_predictions = meta_learner.predict(stacked_test_predictions)
        
        return final_predictions