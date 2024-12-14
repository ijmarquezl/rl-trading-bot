def fetch_historical_prices(symbol, outputsize='full'):
    ...
    params = {
        'function': 'DIGITAL_CURRENCY_DAILY',
        'symbol': symbol,
        'market': 'USD',
        'apikey': 'VBSPQW1YJBWIFRGR',  # Using the API key from .env
        'outputsize': outputsize  # Ensure this is set to 'full'
    }
    if 'Time Series (Digital Currency Daily)' in data:
         print("API Response:", data)  # Print the full response for debugging
         #==================================================================
#
#   File name   : RL-Crypto-Trading-Bot_1.py
#   Author      : IvanML
#   Created Date: 2023-05-26
#   Description : Experiment for creating an autonomous trading Bot
#
#==================================================================

import os
import copy
import random
import numpy as np
import pandas as pd
import pandas_ta as pta
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, Concatenate
from collections import deque
import time
import sys
from dotenv import load_dotenv
import requests

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tensorboardX import SummaryWriter

from model import Actor_Model, Critic_Model
from utils import TradingGraph, Write_to_file
from CryptoTrading import CryptoDataProvider

class CustomEnv:
    # A custom Crypto trading environment
    def __init__(self, df, initial_balance=1000, lookback_window_size=50, Render_range=100):
        self.df = df.dropna().reset_index()
        self.df_total_steps = len(self.df) - 1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.Render_range = Render_range  # render range in visualization
        
        # Action space from 0 to 2, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1, 2])

        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold values for the last lookback_window_size types
        self.orders_history = deque(maxlen=self.lookback_window_size)

        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=self.lookback_window_size)

        # State size contains Market+Orders history for the last lookback_window_size steps
        self.state_size = (self.lookback_window_size, 12)

        # Neural Networks part below
        self.lr = 0.00001
        self.epochs = 1
        self.normalize_value = 100000
        self.optimizer = Adam

        # Create Actor-Critic network model
        self.Actor = Actor_Model(input_shape=self.state_size, action_space=self.action_space.shape[0], lr=self.lr, optimizer=self.optimizer)
        self.Critic = Critic_Model(input_shape=self.state_size, action_space=self.action_space.shape[0], lr=self.lr, optimizer=self.optimizer)

    # create tensorboard writer
    def create_writer(self):
        self.replay_count = 0
        self.writer = SummaryWriter(comment="Crypto_trader")

    # Reset the state of the environment to an initial state
    def reset(self, env_steps_size=0):
        self.visualization = TradingGraph(Render_range=self.Render_range)  # init visualization
        self.trades = deque(maxlen=self.Render_range)  # limited orders memory for visualization

        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.episode_orders = 0  # test
        self.env_steps_size = env_steps_size

        if env_steps_size > 0:  # used for training dataset
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else:  # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps

        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
            self.market_history.append([self.df.loc[current_step, 'Open'],
                                        self.df.loc[current_step, 'High'],
                                        self.df.loc[current_step, 'Low'],
                                        self.df.loc[current_step, 'Close'],
                                        self.df.loc[current_step, 'Volume'],
                                        self.df.loc[self.current_step, 'MACD'],
                                        self.df.loc[self.current_step, 'Signal Line']
                                        ])

        state = np.concatenate((self.market_history, self.orders_history), axis=1)
        return state

    # Get the data points for the given current_step
    def _next_observation(self):
        self.market_history.append([self.df.loc[self.current_step, 'Open'],
                                    self.df.loc[self.current_step, 'High'],
                                    self.df.loc[self.current_step, 'Low'],
                                    self.df.loc[self.current_step, 'Close'],
                                    self.df.loc[self.current_step, 'Volume'],
                                    self.df.loc[self.current_step, 'MACD'],
                                    self.df.loc[self.current_step, 'Signal Line']
                                    ])

        obs = np.concatenate((self.market_history, self.orders_history), axis=1)
        return obs

    # Execute one time step within the environment
    def step(self, action):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        # Set the current price to a random price between open and close
        current_price = random.uniform(
            self.df.loc[self.current_step, 'Open'],
            self.df.loc[self.current_step, 'Close'])
        Date = self.df.loc[self.current_step, 'Date']  # for visualization
        High = self.df.loc[self.current_step, 'High']  # for visualization
        Low = self.df.loc[self.current_step, 'Low']  # for visualization
        MACD = self.df.loc[self.current_step, 'MACD']  # for visualization
        signal = self.df.loc[self.current_step, 'Signal Line']  # for visualization
        RSI = self.df.loc[self.current_step, 'rsi']  # for visualization

        if action == 0:  # Hold
            pass

        elif action == 1 and self.balance > 0:
            # Buy with 100% of current balance
            self.crypto_bought = self.balance / current_price
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought
            self.trades.append({'Date': Date, 'High': High, 'Low': Low, 'MACD': MACD, 'Signal Line': signal, 'RSI': RSI, 'total': self.crypto_bought, 'type': "buy"})
            self.episode_orders += 1

        elif action == 2 and self.crypto_held > 0:
            # Sell 100% of current crypto held
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * current_price
            self.crypto_held -= self.crypto_sold
            self.trades.append({'Date': Date, 'High': High, 'Low': Low, 'MACD': MACD, 'Signal Line': signal, 'RSI': RSI, 'total': self.crypto_sold, 'type': "sell"})
            self.episode_orders += 1

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

        # Calculate reward
        reward = self.net_worth - self.prev_net_worth
        if self.net_worth <= self.initial_balance / 2:
            done = True
        else:
            done = False

        obs = self._next_observation()

        return obs, reward, done

    # Render environment
    def render(self, visualize=False):
        # print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')
        if visualize:
            Date = self.df.loc[self.current_step, 'Date']
            Open = self.df.loc[self.current_step, 'Open']
            Close = self.df.loc[self.current_step, 'Close']
            High = self.df.loc[self.current_step, 'High']
            Low = self.df.loc[self.current_step, 'Low']
            Volume = self.df.loc[self.current_step, 'Volume']
            MACD = self.df.loc[self.current_step, 'MACD']
            signal = self.df.loc[self.current_step, 'Signal Line']
            rsi = self.df.loc[self.current_step, 'rsi']

            # Render the environment to the screen
            self.visualization.render(Date, Open, High, Low, Close, Volume, MACD, signal, rsi, self.net_worth, self.trades)

    def get_gaes(self, rewards, dones, values, next_values, gamma=0.99, lamda=0.95, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Compute discounted rewards
        # discounted_r = np.vstack(self.discount_rewards(rewards))

        # Get Critic network predictions 
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)
        # Compute advantages
        # advantages = discounted_r - values
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
        # stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])
        
        # training Actor and Critic networks
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=True)
        c_loss = self.Critic.Critic.fit(states, target, epochs=self.epochs, verbose=0, shuffle=True)

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        self.replay_count += 1
        
    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict(np.expand_dims(state, axis=0))[0]
        action = np.random.choice(self.action_space, p=prediction)
        return action, prediction

    def save(self, name="Crypto_trader"):
        # save keras model weights
        self.Actor.Actor.save_weights(f"{name}_Actor.weights.h5")
        self.Critic.Critic.save_weights(f"{name}_Critic.weights.h5")

    def load(self, name="Crypto_trader"):
        # load keras model weights
        self.Actor.Actor.load_weights(f"{name}_Actor.weights.h5")
        self.Critic.Critic.load_weights(f"{name}_Critic.weights.h5")

def load_or_fetch_data(symbol, cache_file='data/eth_historical.csv'):
    """
    Loads data from cache if available and recent, otherwise fetches from API.
    Updates cache with new data.
    
    Parameters
    ----------
    symbol : str
        The cryptocurrency symbol (e.g., 'ETH')
    cache_file : str
        Path to the cache file
        
    Returns
    -------
    df : pd.DataFrame
        Historical price data
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        current_date = pd.Timestamp.now()
        cache_exists = os.path.exists(cache_file)
        should_fetch_new = True
        
        if cache_exists:
            # Load cached data
            cached_df = pd.read_csv(cache_file)
            cached_df['Date'] = pd.to_datetime(cached_df['Date'])
            cached_df.set_index('Date', inplace=True)
            
            if len(cached_df) > 0:
                last_date = cached_df.index[-1]
                # If cache is recent (less than 1 day old), use it
                if (current_date - last_date).days < 1:
                    print(f"Using cached data (last updated: {last_date})")
                    should_fetch_new = False
                    return cached_df
                else:
                    print(f"Cache is outdated (last updated: {last_date})")
        
        if should_fetch_new:
            print("Fetching new data from Alpha Vantage...")
            new_df = fetch_historical_prices(symbol)
            
            if new_df is not None and len(new_df) > 0:
                # Save to cache
                new_df.index.name = 'Date'
                new_df.to_csv(cache_file)
                print(f"Data cached successfully to {cache_file}")
                return new_df
            else:
                if cache_exists:
                    print("Failed to fetch new data, using cached data instead")
                    return cached_df
                else:
                    raise ValueError("Failed to fetch data and no cache available")
                    
    except Exception as e:
        print(f"Error in load_or_fetch_data: {e}")
        if cache_exists:
            print("Using cached data due to error")
            return pd.read_csv(cache_file, index_col='Date', parse_dates=True)
        raise

    # Print the columns of the DataFrame to diagnose the KeyError
    print(f'DataFrame columns: {df.columns.tolist()}')

    return df

def fetch_historical_prices(symbol, outputsize='full'):
    csv_file = f'{symbol}_historical_prices.csv'
    # Check if the CSV file exists
    if os.path.exists(csv_file):
        # Load existing data
        df = pd.read_csv(csv_file, index_col='Date', parse_dates=True)
        last_date = df.index[-1]
        print(f'Existing data found. Last date in data: {last_date}')
    else:
        df = pd.DataFrame()
        last_date = None
        print('No existing data found. Fetching full historical data.')

    # Fetch new data only if necessary
    if last_date is None or last_date < pd.Timestamp.now() - pd.DateOffset(days=1):
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'DIGITAL_CURRENCY_DAILY',
            'symbol': symbol,
            'market': 'USD',
            'apikey': 'VBSPQW1YJBWIFRGR',  # Using the API key from .env
            'outputsize': 'compact'
        }
        try:
            response = requests.get(url, params=params)
            data = response.json()
            if 'Time Series (Digital Currency Daily)' in data:
                new_data = pd.DataFrame(data['Time Series (Digital Currency Daily)']).T
                new_data.columns = ['1a. open (USD)', '2a. high (USD)', '3a. low (USD)', '4a. close (USD)', '5. volume', '6. market cap (USD)']
                new_data.index = pd.to_datetime(new_data.index)
                new_data = new_data.astype(float)
                # Combine with existing data
                df = pd.concat([df, new_data])
                df = df[~df.index.duplicated(keep='last')]  # Remove duplicates if any
                df.to_csv(csv_file)
                print(f'Updated data saved to {csv_file}')
            else:
                print('Error fetching data:', data.get('Error Message', 'Unknown error'))
        except Exception as e:
            print(f'Error fetching historical prices: {str(e)}')
            return None
    else:
        print('Data is up to date.')

    return df

def getCurrentPrice(symbol):
    """
    Gets the current price of a cryptocurrency using Alpha Vantage API
    
    Parameters:
    symbol (str): The cryptocurrency symbol (e.g., 'ETH')
    
    Returns:
    float: Current price of the cryptocurrency
    """
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'CRYPTO_QUOTE',
        'symbol': symbol,
        'market': 'USD',
        'apikey': 'VBSPQW1YJBWIFRGR'  # Using the API key from .env
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'Realtime Currency Exchange Rate' in data:
            return float(data['Realtime Currency Exchange Rate']['5. Exchange Rate'])
        else:
            print("Error getting current price:", data.get('Note', 'Unknown error'))
            return None
    except Exception as e:
        print(f"Error fetching current price: {str(e)}")
        return None

def check_trading_signals(df, current_price):
    """
    Check if current market conditions meet our trading criteria
    
    Parameters:
    df (pd.DataFrame): DataFrame with our indicators
    current_price (float): Current price of the asset
    
    Returns:
    str: Trading signal ('buy', 'sell', or 'hold')
    """
    # Get the latest indicators
    latest_macd = df['MACD'].iloc[-1]
    latest_signal = df['Signal Line'].iloc[-1]
    latest_rsi = df['rsi'].iloc[-1]
    
    # Trading conditions
    # Buy conditions:
    # 1. MACD crosses above Signal Line (bullish crossover)
    # 2. RSI is below 70 (not overbought)
    if latest_macd > latest_signal and latest_rsi < 70:
        return 'buy'
    
    # Sell conditions:
    # 1. MACD crosses below Signal Line (bearish crossover)
    # 2. RSI is above 30 (not oversold)
    elif latest_macd < latest_signal and latest_rsi > 30:
        return 'sell'
    
    # If no conditions are met, hold
    return 'hold'

def live_trading(env, symbol='ETH', check_interval=60):
    """
    Performs live trading based on current market conditions
    
    Parameters:
    env: CustomEnv instance
    symbol (str): Trading symbol
    check_interval (int): Seconds between each check
    """
    print(f"Starting live trading for {symbol}...")
    
    while True:
        try:
            # Get current price
            current_price = getCurrentPrice(symbol)
            if current_price is None:
                print("Could not get current price. Waiting for next interval...")
                time.sleep(check_interval)
                continue
            
            # Update our DataFrame with the new price
            new_row = pd.DataFrame({
                '4a. close (USD)': [current_price],
                'Adj Close': [current_price],
                # Add other required columns with appropriate values
            })
            
            # Update indicators
            df = pd.concat([env.df, new_row])
            
            # Recalculate MACD
            ShortEMA = df['4a. close (USD)'].ewm(span=12, adjust=False).mean()
            LongEMA = df['4a. close (USD)'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ShortEMA - LongEMA
            df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Recalculate RSI
            df['rsi'] = pta.rsi(df['Adj Close'], 2)
            
            # Check trading signals
            signal = check_trading_signals(df, current_price)
            
            # Execute trades based on signals
            if signal == 'buy' and env.balance > 0:
                print(f"Buy signal detected at price: ${current_price}")
                # Execute buy using env.step()
                action = 1  # buy action
                obs, reward, done = env.step(action)
                print(f"Executed buy. Balance: ${env.balance:.2f}, Crypto held: {env.crypto_held:.6f}")
                
            elif signal == 'sell' and env.crypto_held > 0:
                print(f"Sell signal detected at price: ${current_price}")
                # Execute sell using env.step()
                action = 2  # sell action
                obs, reward, done = env.step(action)
                print(f"Executed sell. Balance: ${env.balance:.2f}, Crypto held: {env.crypto_held:.6f}")
            
            print(f"Current price: ${current_price:.2f}")
            print(f"MACD: {df['MACD'].iloc[-1]:.6f}")
            print(f"Signal Line: {df['Signal Line'].iloc[-1]:.6f}")
            print(f"RSI: {df['rsi'].iloc[-1]:.2f}")
            print("-" * 50)
            
            # Wait for next interval
            time.sleep(check_interval)
            
        except Exception as e:
            print(f"Error in live trading: {str(e)}")
            time.sleep(check_interval)

def Random_games(env, visualize, train_episodes = 50):
    average_net_worth = 0
    for episode in range(train_episodes):
        state = env.reset()
        while True:
            env.render(visualize)
            action = np.random.randint(3, size=1)[0]
            state, reward, done = env.step(action)
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", env.net_worth)
                break

    # print("average_net_worth:", average_net_worth/train_episodes)
    print(f'average {train_episodes} episodes random net_worth: {average_net_worth/train_episodes}')

def train_agent(env, visualize=False, train_episodes=20, training_batch_size=4, checkpoint_interval=10):
    env.create_writer() # create TensorBoard writer
    total_average = deque(maxlen=100) # save recent 100 episodes net worth
    best_average = 0 # used to track best average net worth
    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)

        states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []
        for t in range(training_batch_size):
            env.render(visualize)
            action, prediction = env.act(state)
            next_state, reward, done = env.step(action)
            states.append(np.expand_dims(state, axis=0))
            next_states.append(np.expand_dims(next_state, axis=0))
            action_onehot = np.zeros(3)
            action_onehot[action] = 1
            actions.append(action_onehot)
            rewards.append(reward)
            dones.append(done)
            predictions.append(prediction)
            state = next_state

            if episode % checkpoint_interval == 0 and episode > 0:  
                print("Saving checkpoint")
                env.Actor.save()
                env.Critic.save()

    env.replay(states, actions, rewards, predictions, dones, next_states)
    total_average.append(env.net_worth)
    average = np.average(total_average)

    env.writer.add_scalar('Data/average net_worth', average, episode)
    env.writer.add_scalar('Data/episode_orders', env.episode_orders, episode)

    print("net worth {} {:.2f} {:.2f} {}".format(episode, env.net_worth, average, env.episode_orders))
    if episode > len(total_average):
        if best_average < average:
            best_average = average
            print("Saving model")
            env.save()

def test_agent(env, visualize=True, test_episodes=10):
    env.load() # Load the model
    average_net_worth = 0
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize)
            action, prediction = env.act(state)
            state, reward, done =env.step(action)
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth: ", episode, env.net_worth, env.episode_orders)
                break

    print("average {} episodes agent net_worth: {}".format(test_episodes, average_net_worth/test_episodes))

def resume_training(env, visualize=False, train_episodes=50, training_batch_size=500, checkpoint_path="Crypto_trader_checkpoint"):
    env.Actor.load(checkpoint_path)  
    env.Critic.load(checkpoint_path)
    train_agent(env, visualize, train_episodes, training_batch_size)

def Play_games(env, visualize):
    average_net_worth = 0
    state = env.reset()
    # while True:
    env.render(visualize)
    action, prediction = env.act(state)
    state, reward, done = env.step(action)
    # if env.current_step == env.end_step:
    average_net_worth += env.net_worth
    # print("net_worth:", env.net_worth)
    # break

    # print("average_net_worth:", average_net_worth/train_episodes)
    print(f'average net_worth: {average_net_worth}')

def create_actor_model(state_dim, action_dim):
    """
    Create an actor model for the reinforcement learning agent.
    
    Parameters
    ----------
    state_dim : int
        Dimension of the input state
    action_dim : int
        Dimension of the action space
    
    Returns
    -------
    model : keras.Model
        Compiled neural network model
    """
    model = Sequential([
        Dense(64, activation='relu', input_dim=state_dim),
        Dense(32, activation='relu'),
        Dense(action_dim, activation='tanh')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def create_critic_model(state_dim, action_dim):
    """
    Create a critic model for the reinforcement learning agent.
    
    Parameters
    ----------
    state_dim : int
        Dimension of the input state
    action_dim : int
        Dimension of the action space
    
    Returns
    -------
    model : keras.Model
        Compiled neural network model
    """
    state_input = Input(shape=(state_dim,))
    action_input = Input(shape=(action_dim,))
    
    x = Dense(64, activation='relu')(state_input)
    x = Concatenate()([x, action_input])
    x = Dense(32, activation='relu')(x)
    output = Dense(1)(x)
    
    model = Model([state_input, action_input], output)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# Download the historic prices of the asset
print("Downloading ETH historical data...")
df = load_or_fetch_data('ETH')

if df is None or len(df) == 0:
    print("Failed to fetch data from Alpha Vantage. Check your API key and try again.")
    sys.exit(1)

print(f"Successfully loaded {len(df)} days of ETH data")
print(f"Date range: from {df.index[0]} to {df.index[-1]}")

# Calculate technical indicators
print("Calculating technical indicators...")

# Calculate MACD
exp1 = df['4a. close (USD)'].ewm(span=12, adjust=False).mean()
exp2 = df['4a. close (USD)'].ewm(span=26, adjust=False).mean()
MACD = exp1 - exp2
signal = MACD.ewm(span=9, adjust=False).mean()

df['MACD'] = MACD
df['Signal Line'] = signal

# Calculate RSI
df['rsi'] = pta.rsi(df['4a. close (USD)'], 14)  # Using 14-day RSI

# Drop any rows with NaN values
df = df.dropna()

print("Total dataset size:", len(df))

# Split into training and testing sets
lookback_window_size = 5  
train_size = int(len(df) * 0.8)  # Use 80% for training
train_df = df[:train_size]
test_df = df[train_size-lookback_window_size:]  # Overlap by lookback_window_size to ensure continuity

print("Training set size:", len(train_df))
print("Test set size:", len(test_df))

# Dynamically calculate state and action dimensions
state_dim = 5 * lookback_window_size + 4  # market history + portfolio info
action_dim = 3  # Buy, Sell, Hold

# Initialize environments with appropriate initial balance for ETH
initial_balance = 1000  # Starting with $1000
train_env = CustomEnv(train_df, lookback_window_size=lookback_window_size, initial_balance=initial_balance)
test_env = CustomEnv(test_df, lookback_window_size=lookback_window_size, initial_balance=initial_balance)

# Train the agent
training_batch_size = 4    
train_episodes = 20       

print("\nStarting training...")
train_agent(train_env, visualize=False, train_episodes=train_episodes, training_batch_size=training_batch_size)

# Test the trained agent
print("\nTesting the trained agent...")
test_agent(test_env, visualize=True, test_episodes=10)