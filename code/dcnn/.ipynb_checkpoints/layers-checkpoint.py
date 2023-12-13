import tensorflow as tf

from keras import backend as K
from keras.layers import Flatten, Lambda, Layer, Reshape


class SemiDynamicKMaxPooling(Layer):
    """
    Semy-dynamic k-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    K is dynamically computed based on number of conv layers, current conv layer number, and average input sentence length.
    """
    def __init__(self, k_top, L, l, avg_s, **kwargs):
        super().__init__(**kwargs)
        self.k_top = k_top
        self.L = L
        self.l = l
        self.avg_s = avg_s

    def compute_output_shape(self, input_shape):
        #s = input_shape[1]
        k_l = max(self.k_top, int(round((self.L - self.l) / self.L * self.avg_s)))
        return (input_shape[0], (input_shape[2] * self.avg_s))

    def call(self, inputs):
        #s = tf.shape(inputs)[1] 
        
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])
        
        # compute dynamic k value
        k_l = tf.maximum(self.k_top, tf.cast(tf.round((self.L - self.l) / self.L * tf.cast(self.avg_s, tf.float32)), tf.int32))
        # extract top_k values and their indices
        values, indices = tf.nn.top_k(shifted_input, k=k_l, sorted=True)
        
        # sort the indices to maintain the original order
        sorted_indices = tf.argsort(indices, axis=-1)
        batch_indices = tf.broadcast_to(tf.range(tf.shape(indices)[0])[:, tf.newaxis, tf.newaxis], tf.shape(indices))
        channel_indices = tf.broadcast_to(tf.range(tf.shape(indices)[1])[tf.newaxis, :, tf.newaxis], tf.shape(indices))
        
        # use the sorted indices to get the values in their original order
        top_k_ordered = tf.gather_nd(values, tf.stack([batch_indices, channel_indices, sorted_indices], axis=-1))
        
        # return flattened output
        return Flatten()(top_k_ordered)
    
    
class Folding(Layer):
    """Fold a 2D tensor along the row axis."""
    def __init__(self, **kwargs):
        super(Folding, self).__init__(**kwargs)

    def call(self, inputs):
        # Get the shape of the tensor
        _, num_rows, num_features = inputs.shape.as_list()

        # Ensure that the number of rows can be halved
        assert num_rows % 2 == 0, "Number of rows should be even to perform folding."

        # Reshape the tensor to group every two rows together
        reshaped = Reshape((num_rows // 2, 2, num_features))(inputs)

        # Sum along the new axis to fold
        folded = Lambda(lambda x: K.sum(x, axis=2))(reshaped)

        return folded

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 2, input_shape[2])