import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

# Simplified market simulation classes for historical data training
class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Trade:
    side: Side
    qty: int
    price: float
    agent_id: str

class HistoricalMarketSimulator:
    """Simulates market using historical S&P 500 data"""
    
    def __init__(self, data_file_path: str):
        self.df = self.load_and_prepare_data(data_file_path)
        self.current_index = 0
        self.total_ticks = len(self.df)
        
    def load_and_prepare_data(self, file_path: str) -> pd.DataFrame:
        """Load S&P 500 data from Excel file"""
        try:
            # Try different sheet names and formats
            df = None
            sheet_names = [0, 'Sheet1', 'Data', 'SP500', 'S&P500']
            
            for sheet in sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet)
                    break
                except:
                    continue
            
            if df is None:
                df = pd.read_csv(file_path)  # Fallback to CSV
                
            # Standardize column names (handle different formats)
            df.columns = df.columns.str.lower().str.strip()
            
            # Map common column variations
            column_mapping = {
                'date': ['date', 'timestamp', 'time', 'datetime'],
                'open': ['open', 'opening', 'open_price'],
                'high': ['high', 'high_price', 'maximum'],
                'low': ['low', 'low_price', 'minimum'], 
                'close': ['close', 'closing', 'close_price', 'price'],
                'volume': ['volume', 'vol', 'trading_volume']
            }
            
            final_columns = {}
            for target_col, variations in column_mapping.items():
                for var in variations:
                    if var in df.columns:
                        final_columns[var] = target_col
                        break
                        
            df = df.rename(columns=final_columns)
            
            # Ensure we have required columns
            required = ['open', 'high', 'low', 'close']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
                
            # Parse date if exists
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                
            # Calculate additional features
            df['mid_price'] = (df['high'] + df['low']) / 2
            df['spread_pct'] = (df['high'] - df['low']) / df['close']
            df['volatility'] = df['close'].pct_change().rolling(20).std()
            df['returns'] = df['close'].pct_change()
            
            # Remove NaN rows
            df = df.dropna().reset_index(drop=True)
            
            print(f"Loaded {len(df)} data points")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "No date column")
            print(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")
    
    def get_current_market_data(self) -> Dict:
        """Get current market data point"""
        if self.current_index >= self.total_ticks:
            return None
            
        row = self.df.iloc[self.current_index]
        
        # Simulate bid-ask spread (typically 0.01-0.05% for S&P 500)
        spread_pct = max(0.0001, row['spread_pct'] * 0.1)  # Use 10% of daily range as spread
        mid_price = row['close']
        spread = mid_price * spread_pct
        
        return {
            'best_bid': mid_price - spread/2,
            'best_ask': mid_price + spread/2,
            'mid_price': mid_price,
            'volume': row.get('volume', 1000000),
            'spread_pct': spread_pct,
            'volatility': row.get('volatility', 0.01),
            'returns': row.get('returns', 0),
            'timestamp': row.get('date', self.current_index)
        }
    
    def execute_market_order(self, side: Side, qty: int, agent_id: str) -> List[Trade]:
        """Execute a market order at current prices"""
        market_data = self.get_current_market_data()
        if not market_data:
            return []
            
        if side == Side.BUY:
            price = market_data['best_ask']
        else:
            price = market_data['best_bid']
            
        return [Trade(side=side, qty=qty, price=price, agent_id=agent_id)]
    
    def step(self):
        """Move to next time step"""
        self.current_index += 1
        
    def reset(self):
        """Reset to beginning"""
        self.current_index = 0
        
    def is_done(self) -> bool:
        """Check if simulation is complete"""
        return self.current_index >= self.total_ticks - 1

class SP500RLAgent:
    """Reinforcement Learning Agent adapted for S&P 500 trading"""
    
    def __init__(self, agent_id, learning_rate=0.1, discount_factor=0.95, epsilon=0.3, epsilon_decay=0.999):
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.05
        
        # Actions: hold, buy_small, sell_small, buy_large, sell_large
        self.actions = ['hold', 'buy_small', 'sell_small', 'buy_large', 'sell_large']
        self.n_actions = len(self.actions)
        
        # Q-table for state-action values
        self.q_table = {}
        
        # Agent state
        self.inventory = 0
        self.cash = 100000
        self.initial_cash = 100000
        self.max_position = 1000  # Maximum shares to hold
        self.trade_count = 0
        
        # History tracking
        self.price_history = deque(maxlen=50)
        self.portfolio_history = []
        self.reward_history = []
        
        # Learning tracking
        self.last_state = None
        self.last_action = None
        self.last_portfolio_value = self.initial_cash
        
        # Trading constraints
        self.min_trade_interval = 1  # Minimum steps between trades
        self.last_trade_step = -10
        
        # Performance tracking
        self.total_pnl = 0
        self.max_drawdown = 0
        self.peak_portfolio_value = self.initial_cash
        
    def get_state(self, market_data: Dict, step: int) -> tuple:
        """Extract state features from market data"""
        if not market_data:
            return (1, 1, 1, 1)  # Neutral state
            
        mid_price = market_data['mid_price']
        self.price_history.append(mid_price)
        
        # 1. Price trend (short-term momentum)
        if len(self.price_history) >= 10:
            short_trend = (self.price_history[-1] - self.price_history[-5]) / self.price_history[-5]
            if short_trend < -0.005:  # Down > 0.5%
                trend = 0
            elif short_trend > 0.005:  # Up > 0.5%
                trend = 2
            else:
                trend = 1  # Flat
        else:
            trend = 1
            
        # 2. Longer trend (momentum)
        if len(self.price_history) >= 20:
            long_trend = (self.price_history[-1] - self.price_history[-20]) / self.price_history[-20]
            if long_trend < -0.02:  # Down > 2%
                momentum = 0
            elif long_trend > 0.02:  # Up > 2%
                momentum = 2
            else:
                momentum = 1
        else:
            momentum = 1
            
        # 3. Current position state
        position_ratio = self.inventory / self.max_position
        if position_ratio < -0.5:
            position = 0  # Heavy short
        elif position_ratio > 0.5:
            position = 2  # Heavy long
        else:
            position = 1  # Neutral
            
        # 4. Market volatility/spread
        spread_pct = market_data.get('spread_pct', 0.001)
        if spread_pct > 0.002:  # High spread
            volatility = 2
        elif spread_pct < 0.0005:  # Low spread
            volatility = 0
        else:
            volatility = 1
            
        return (trend, momentum, position, volatility)
    
    def choose_action(self, state: tuple, step: int) -> int:
        """Choose action using epsilon-greedy policy"""
        # Prevent overtrading
        if step - self.last_trade_step < self.min_trade_interval:
            return 0  # Hold
            
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        # Get Q-values for current state
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        
        return np.argmax(self.q_table[state])
    
    def execute_action(self, action: int, market_simulator: HistoricalMarketSimulator, step: int) -> List[Trade]:
        """Execute the chosen action"""
        market_data = market_simulator.get_current_market_data()
        if not market_data:
            return []
            
        trades = []
        
        # Define trade sizes
        small_size = 50  # 50 shares
        large_size = 200  # 200 shares
        
        # Position limits check
        if action in [1, 3] and self.inventory >= self.max_position:  # Buy actions
            return []
        if action in [2, 4] and self.inventory <= -self.max_position:  # Sell actions
            return []
        
        # Execute actions
        if action == 0:  # Hold
            pass
        elif action == 1:  # Buy small
            trades = market_simulator.execute_market_order(Side.BUY, small_size, self.agent_id)
        elif action == 2:  # Sell small
            trades = market_simulator.execute_market_order(Side.SELL, small_size, self.agent_id)
        elif action == 3:  # Buy large
            trades = market_simulator.execute_market_order(Side.BUY, large_size, self.agent_id)
        elif action == 4:  # Sell large
            trades = market_simulator.execute_market_order(Side.SELL, large_size, self.agent_id)
        
        # Update agent state
        for trade in trades:
            if trade.side == Side.BUY:
                self.inventory += trade.qty
                self.cash -= trade.price * trade.qty
            else:
                self.inventory -= trade.qty
                self.cash += trade.price * trade.qty
                
            self.trade_count += 1
            self.last_trade_step = step
        
        return trades
    
    def calculate_reward(self, market_data: Dict, trades: List[Trade], action: int) -> float:
        """Calculate reward for the action taken"""
        current_portfolio_value = self.get_portfolio_value(market_data['mid_price'])
        
        # Basic P&L reward
        pnl_change = current_portfolio_value - self.last_portfolio_value
        reward = pnl_change / 1000  # Scale down the reward
        
        # Transaction cost penalty
        if trades:
            transaction_cost = len(trades) * market_data['mid_price'] * 0.001  # 0.1% transaction cost
            reward -= transaction_cost / 1000
        
        # Position risk penalty
        position_risk = abs(self.inventory) / self.max_position
        if position_risk > 0.8:
            reward -= position_risk * 0.5
        
        # Holding reward (encourage patience)
        if action == 0 and abs(self.inventory) < self.max_position * 0.2:
            reward += 0.01
        
        self.last_portfolio_value = current_portfolio_value
        self.reward_history.append(reward)
        
        return reward
    
    def update_q_table(self, old_state: tuple, action: int, reward: float, new_state: tuple):
        """Update Q-table using Q-learning"""
        if old_state not in self.q_table:
            self.q_table[old_state] = np.zeros(self.n_actions)
        if new_state not in self.q_table:
            self.q_table[new_state] = np.zeros(self.n_actions)
        
        # Q-learning update
        old_q = self.q_table[old_state][action]
        max_future_q = np.max(self.q_table[new_state])
        new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_future_q - old_q)
        self.q_table[old_state][action] = new_q
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value"""
        return self.cash + self.inventory * current_price
    
    def get_stats(self, current_price: float) -> Dict:
        """Get current agent statistics"""
        portfolio_value = self.get_portfolio_value(current_price)
        total_pnl = portfolio_value - self.initial_cash
        
        # Update tracking
        self.portfolio_history.append(portfolio_value)
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
        
        current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        return {
            'portfolio_value': portfolio_value,
            'total_pnl': total_pnl,
            'inventory': self.inventory,
            'cash': self.cash,
            'trade_count': self.trade_count,
            'epsilon': self.epsilon,
            'avg_reward': np.mean(self.reward_history[-100:]) if self.reward_history else 0,
            'max_drawdown': self.max_drawdown,
            'q_table_size': len(self.q_table)
        }

def train_agent_on_sp500(data_file_path: str, episodes: int = 100, plot_results: bool = True):
    """Train the RL agent on S&P 500 historical data"""
    
    print("Loading S&P 500 data...")
    market_sim = HistoricalMarketSimulator(data_file_path)
    
    print("Initializing RL agent...")
    agent = SP500RLAgent("RL_Agent_SP500")
    
    # Training tracking
    episode_returns = []
    episode_trades = []
    episode_portfolios = []
    
    print(f"Starting training for {episodes} episodes...")
    
    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")
        
        # Reset environment
        market_sim.reset()
        agent.inventory = 0
        agent.cash = agent.initial_cash
        agent.last_portfolio_value = agent.initial_cash
        agent.trade_count = 0
        agent.price_history.clear()
        
        episode_rewards = []
        step = 0
        
        # Run episode
        while not market_sim.is_done():
            # Get current market data and state
            market_data = market_sim.get_current_market_data()
            if not market_data:
                break
                
            current_state = agent.get_state(market_data, step)
            
            # Choose and execute action
            action = agent.choose_action(current_state, step)
            trades = agent.execute_action(action, market_sim, step)
            
            # Calculate reward
            reward = agent.calculate_reward(market_data, trades, action)
            episode_rewards.append(reward)
            
            # Move to next step
            market_sim.step()
            next_market_data = market_sim.get_current_market_data()
            
            if next_market_data:
                next_state = agent.get_state(next_market_data, step + 1)
                
                # Update Q-table
                if agent.last_state is not None:
                    agent.update_q_table(agent.last_state, agent.last_action, reward, current_state)
                
                agent.last_state = current_state
                agent.last_action = action
            
            step += 1
            
            # Progress update
            if step % 1000 == 0:
                stats = agent.get_stats(market_data['mid_price'])
                print(f"  Step {step}: Portfolio=${stats['portfolio_value']:.2f}, "
                      f"PnL=${stats['total_pnl']:.2f}, Trades={stats['trade_count']}")
        
        # Episode summary
        final_market_data = market_sim.df.iloc[-1]
        final_stats = agent.get_stats(final_market_data['close'])
        
        episode_returns.append(final_stats['total_pnl'])
        episode_trades.append(final_stats['trade_count'])
        episode_portfolios.append(final_stats['portfolio_value'])
        
        print(f"Episode {episode + 1} complete:")
        print(f"  Final P&L: ${final_stats['total_pnl']:.2f}")
        print(f"  Total trades: {final_stats['trade_count']}")
        print(f"  Epsilon: {final_stats['epsilon']:.3f}")
        print(f"  Max drawdown: {final_stats['max_drawdown']*100:.2f}%")
        print("-" * 50)
    
    if plot_results:
        plot_training_results(episode_returns, episode_trades, episode_portfolios)
    
    return agent, {
        'episode_returns': episode_returns,
        'episode_trades': episode_trades, 
        'episode_portfolios': episode_portfolios,
        'market_data': market_sim.df
    }

def plot_training_results(episode_returns, episode_trades, episode_portfolios):
    """Plot training results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # P&L over episodes
    axes[0,0].plot(episode_returns)
    axes[0,0].set_title('Total P&L per Episode')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('P&L ($)')
    axes[0,0].grid(True)
    
    # Portfolio value over episodes
    axes[0,1].plot(episode_portfolios)
    axes[0,1].axhline(y=100000, color='r', linestyle='--', label='Initial Value')
    axes[0,1].set_title('Final Portfolio Value per Episode')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Portfolio Value ($)')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Trades per episode
    axes[1,0].plot(episode_trades)
    axes[1,0].set_title('Number of Trades per Episode')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Trade Count')
    axes[1,0].grid(True)
    
    # Rolling average P&L
    window = min(10, len(episode_returns))
    rolling_pnl = pd.Series(episode_returns).rolling(window=window).mean()
    axes[1,1].plot(rolling_pnl)
    axes[1,1].set_title(f'Rolling Average P&L ({window} episodes)')
    axes[1,1].set_xlabel('Episode')
    axes[1,1].set_ylabel('Avg P&L ($)')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()

# MAIN EXECUTION
if __name__ == "__main__":
    # Your Excel file
    excel_file_path = "SP500_TrainData.xlsx"
    
    try:
        print("Starting S&P 500 RL Agent Training...")
        print(f"Using data file: {excel_file_path}")
        
        # Train the agent
        trained_agent, results = train_agent_on_sp500(
            data_file_path=excel_file_path,
            episodes=50,  # Start with 50 episodes for testing
            plot_results=True
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final Q-table size: {len(trained_agent.q_table)} states learned")
        print(f"Best episode return: ${max(results['episode_returns']):.2f}")
        print(f"Worst episode return: ${min(results['episode_returns']):.2f}")
        print(f"Average return: ${np.mean(results['episode_returns']):.2f}")
        print(f"Standard deviation: ${np.std(results['episode_returns']):.2f}")
        
        # Additional statistics
        positive_episodes = sum(1 for r in results['episode_returns'] if r > 0)
        success_rate = positive_episodes / len(results['episode_returns']) * 100
        print(f"Profitable episodes: {positive_episodes}/{len(results['episode_returns'])} ({success_rate:.1f}%)")
        
        print(f"\nFinal agent epsilon (exploration rate): {trained_agent.epsilon:.4f}")
        print(f"Total unique states explored: {len(trained_agent.q_table)}")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find the file '{excel_file_path}'")
        print("Please make sure:")
        print("1. The file 'SP500_TrainData.xlsx' is in the same directory as this script")
        print("2. The filename is spelled exactly right (case-sensitive)")
        
    except ValueError as e:
        print(f"❌ Data format error: {e}")
        print("\nYour Excel file should have these columns:")
        print("Required: Open, High, Low, Close")
        print("Optional: Date, Volume")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Please check your Excel file format and data quality")