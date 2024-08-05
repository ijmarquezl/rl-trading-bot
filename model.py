#================================================================
#
#   File name   : model.py
#   Author      : IvanML
#   Created date: 2024-05-27
#   Description : defined PPO Keras model Actor and Critic classes
#
#================================================================
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras import backend as K
tf.config.experimental_run_functions_eagerly(True) # used for debuging and development
#tf.compat.v1.disable_eager_execution() # usually using this for fastest performance

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass

class Actor_Model:
    """
    Initializes an Actor_Model object with the given input shape, action space, learning rate, and optimizer.

    Parameters:
        input_shape (tuple): The shape of the input data.
        action_space (int): The number of possible actions.
        lr (float): The learning rate.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer to use for training.

    Returns:
        None
    """
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        self.action_space = action_space

        X = Flatten(input_shape=input_shape)(X_input)
        X = Dense(512, activation="relu")(X)
        X = Dense(256, activation="relu")(X)
        X = Dense(64, activation="relu")(X)
        output = Dense(self.action_space, activation="softmax")(X)

        self.Actor = Model(inputs = X_input, outputs = output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=lr))

        """
        Initializes an Actor_Model object with the given input shape, action space, learning rate, and optimizer.
        
        Parameters:
            y_true (tensor): The true values.
            y_pred (tensor): The predicted values.
            
        Returns:
            total_loss (tensor): The calculated total loss.
        """
    def ppo_loss(self, y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))
        
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        
        total_loss = actor_loss - entropy

        return total_loss

        """
        Save the model checkpoint to the specified path.

        Args:
            checkpoint_path (str, optional): The path where the checkpoint will be saved. Defaults to "Crypto_trader_checkpoint".

        Returns:
            None
        """
    def save(self, checkpoint_path="Crypto_trader_checkpoint"):
        # Create a checkpoint object
        checkpoint = tf.train.Checkpoint(model=self.Actor) 
        # Save the checkpoint
        checkpoint.save(file_prefix=checkpoint_path)

        """
        Load the model checkpoint from the specified path.

        Args:
            checkpoint_path (str, optional): The path where the checkpoint is saved. Defaults to "Crypto_trader_checkpoint".

        Returns:
            None

        Raises:
            None

        This function creates a checkpoint object using the `tf.train.Checkpoint` class and saves it with the specified model.
        It then restores the latest checkpoint from the specified path using the `tf.train.latest_checkpoint` function.
        If a checkpoint is found, it is restored and a message is printed indicating the checkpoint loaded.
        If no checkpoint is found, a message is printed indicating that no checkpoint was found.
        """
    def load(self, checkpoint_path="Crypto_trader_checkpoint"):
        # Create a checkpoint object
        checkpoint = tf.train.Checkpoint(model=self.Actor)
        # Restore the latest checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            print("Checkpoint loaded from:", latest_checkpoint)
        else:
            print("No checkpoint found.")

    def predict(self, state):
        return self.Actor.predict(state)

class Critic_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)

        V = Flatten(input_shape=input_shape)(X_input)
        V = Dense(512, activation="relu")(V)
        V = Dense(256, activation="relu")(V)
        V = Dense(64, activation="relu")(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=X_input, outputs = value)
        self.Critic.compile(loss=self.critic_PPO2_loss, optimizer=optimizer(learning_rate=lr))

    def critic_PPO2_loss(self, y_true, y_pred):
        value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss

    def save(self, checkpoint_path="Crypto_trader_checkpoint"):
        # Create a checkpoint object
        checkpoint = tf.train.Checkpoint(model=self.Critic) 
        # Save the checkpoint
        checkpoint.save(file_prefix=checkpoint_path)

    def load(self, checkpoint_path="Crypto_trader_checkpoint"):
        # Create a checkpoint object
        checkpoint = tf.train.Checkpoint(model=self.Critic)
        # Restore the latest checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            print("Checkpoint loaded from:", latest_checkpoint)
        else:
            print("No checkpoint found.")

    def predict(self, state):
        # return self.Critic.predict([state, np.zeros((state.shape[0], 1))])
        return self.Critic.predict(state)