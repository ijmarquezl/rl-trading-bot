import web3
from web3 import Web3
import requests
import json
import os
from dotenv import load_dotenv
import pandas as pd
import pandas_ta as pta
from datetime import datetime
import time

class CryptoTrader:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize Web3 and connect to Ethereum network
        self.infura_url = f"https://mainnet.infura.io/v3/{os.getenv('INFURA_PROJECT_ID')}"
        self.w3 = Web3(Web3.HTTPProvider(self.infura_url))
        
        # MetaMask wallet configuration
        self.account_address = os.getenv('METAMASK_ADDRESS')
        self.private_key = os.getenv('PRIVATE_KEY')
        
        # Alpha Vantage API configuration
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY')
        
        # Verify connection
        if not self.w3.is_connected():
            raise Exception("Failed to connect to Ethereum network")
            
        # Initialize price cache
        self.price_cache = {}
        self.last_update = None
        
    def get_current_price(self, symbol):
        """
        Gets the current price of a cryptocurrency using Alpha Vantage API
        """
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'CRYPTO_QUOTE',
            'symbol': symbol,
            'market': 'USD',
            'apikey': self.alpha_vantage_key
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
            
    def get_historical_prices(self, symbol, interval='60min', start_date=None, end_date=None):
        """
        Gets historical price data for a cryptocurrency
        """
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'apikey': self.alpha_vantage_key,
            'outputsize': 'full'
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            df = pd.DataFrame(data[f'Time Series ({interval})']).T
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"Error fetching historical prices: {str(e)}")
            return None
            
    def calculate_indicators(self, df):
        """
        Calculates technical indicators (MACD and RSI)
        """
        # Calculate MACD
        short_ema = df['Close'].ewm(span=12, adjust=False).mean()
        long_ema = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = short_ema - long_ema
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Calculate RSI
        df['RSI'] = pta.rsi(df['Close'], length=14)
        
        return df
        
    def check_trading_signals(self, df):
        """
        Analyzes current market conditions and returns trading signals
        """
        latest_macd = df['MACD'].iloc[-1]
        latest_signal = df['Signal_Line'].iloc[-1]
        latest_rsi = df['RSI'].iloc[-1]
        
        # Trading conditions
        if latest_macd > latest_signal and latest_rsi < 70:
            return 'buy'
        elif latest_macd < latest_signal and latest_rsi > 30:
            return 'sell'
        return 'hold'
        
    def get_eth_balance(self):
        """
        Gets the ETH balance of the MetaMask wallet
        """
        try:
            balance_wei = self.w3.eth.get_balance(self.account_address)
            balance_eth = self.w3.from_wei(balance_wei, 'ether')
            return balance_eth
        except Exception as e:
            print(f"Error getting ETH balance: {str(e)}")
            return None
            
    def execute_trade(self, action, amount_eth, to_address=None):
        """
        Executes a trade using MetaMask wallet
        """
        try:
            if action not in ['buy', 'sell']:
                raise ValueError("Invalid action. Must be 'buy' or 'sell'")
                
            nonce = self.w3.eth.get_transaction_count(self.account_address)
            
            # Convert ETH amount to Wei
            amount_wei = self.w3.to_wei(amount_eth, 'ether')
            
            # Prepare transaction
            transaction = {
                'nonce': nonce,
                'to': to_address if to_address else self.account_address,
                'value': amount_wei,
                'gas': 21000,
                'gasPrice': self.w3.eth.gas_price,
                'chainId': 1  # mainnet
            }
            
            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(
                transaction, private_key=self.private_key
            )
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for transaction receipt
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            print(f"{action.capitalize()} transaction successful!")
            print(f"Transaction hash: {tx_receipt.transactionHash.hex()}")
            return tx_receipt
            
        except Exception as e:
            print(f"Error executing trade: {str(e)}")
            return None
            
    def start_live_trading(self, symbol='ETH', check_interval=60):
        """
        Starts live trading with automatic signal detection and execution
        """
        print(f"Starting live trading for {symbol}...")
        
        while True:
            try:
                # Get current price
                current_price = self.get_current_price(symbol)
                if not current_price:
                    continue
                    
                # Get recent historical data
                df = self.get_historical_prices(symbol)
                if df is None:
                    continue
                    
                # Calculate indicators
                df = self.calculate_indicators(df)
                
                # Check trading signals
                signal = self.check_trading_signals(df)
                
                # Get current balance
                eth_balance = self.get_eth_balance()
                
                print(f"\nCurrent price: ${current_price:.2f}")
                print(f"Current balance: {eth_balance:.4f} ETH")
                print(f"Signal: {signal}")
                
                # Execute trades based on signals
                if signal == 'buy' and eth_balance > 0.1:  # Ensure minimum balance for gas
                    trade_amount = eth_balance * 0.5  # Trade with 50% of balance
                    self.execute_trade('buy', trade_amount)
                    
                elif signal == 'sell' and eth_balance > 0.1:
                    trade_amount = eth_balance * 0.5
                    self.execute_trade('sell', trade_amount)
                    
                # Wait for next interval
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"Error in live trading loop: {str(e)}")
                time.sleep(check_interval)
                
if __name__ == "__main__":
    # Example usage
    trader = CryptoTrader()
    
    # Start live trading
    trader.start_live_trading(symbol='ETH', check_interval=60)