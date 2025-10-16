import random
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from OrderBook import OrderBook, Order, Side, OrderType

# Agent RL amélioré avec Deep Q-Network simplifié
class ImprovedRLAgent:
    def __init__(self, agent_id, learning_rate=0.01, discount_factor=0.99, epsilon=0.3, epsilon_decay=0.995):
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01
        
        # États plus détaillés et actions plus réalistes
        self.actions = ['hold', 'buy_aggressive', 'sell_aggressive', 'buy_passive', 'sell_passive']
        self.n_actions = len(self.actions)
        
        # Table Q avec états plus complexes
        self.q_table = {}
        
        # État de l'agent
        self.inventory = 0
        self.cash = 100000
        self.initial_cash = 100000
        self.position_limit = 200
        self.max_order_size = 100
        
        # Métriques de performance
        self.total_pnl = 0
        self.realized_pnl = 0
        self.trade_count = 0
        self.profitable_trades = 0
        
        # Historiques pour l'apprentissage
        self.price_history = deque(maxlen=10)
        self.reward_history = deque(maxlen=100)
        self.action_history = []
        self.last_state = None
        self.last_action = None
        
        # Métriques d'apprentissage
        self.exploration_count = 0
        self.exploitation_count = 0
        
    def get_state(self, order_book, tick):
        """État plus riche pour un meilleur apprentissage"""
        snapshot = order_book.snapshot()
        best_bid = self._get_best_price(snapshot, 'bid')
        best_ask = self._get_best_price(snapshot, 'ask')
        
        if best_bid is None or best_ask is None:
            return (1, 1, 1, 1)  # État neutre par défaut
        
        mid_price = (best_bid + best_ask) / 2
        self.price_history.append(mid_price)
        
        # 1. Spread relatif (0=très serré, 1=serré, 2=normal, 3=large)
        spread = best_ask - best_bid
        spread_bps = (spread / mid_price) * 10000  # en basis points
        if spread_bps < 5:
            spread_level = 0
        elif spread_bps < 15:
            spread_level = 1
        elif spread_bps < 30:
            spread_level = 2
        else:
            spread_level = 3
            
        # 2. Momentum des prix (0=forte baisse, 1=baisse, 2=stable, 3=hausse, 4=forte hausse)
        if len(self.price_history) >= 5:
            recent_change = (self.price_history[-1] - self.price_history[-5]) / self.price_history[-5]
            if recent_change < -0.002:
                momentum = 0
            elif recent_change < -0.0005:
                momentum = 1
            elif recent_change < 0.0005:
                momentum = 2
            elif recent_change < 0.002:
                momentum = 3
            else:
                momentum = 4
        else:
            momentum = 2
            
        # 3. Position relative (0=très short, 1=short, 2=neutre, 3=long, 4=très long)
        if self.inventory < -150:
            position_level = 0
        elif self.inventory < -50:
            position_level = 1
        elif self.inventory < 50:
            position_level = 2
        elif self.inventory < 150:
            position_level = 3
        else:
            position_level = 4
            
        # 4. Performance récente (0=mauvaise, 1=neutre, 2=bonne)
        recent_rewards = list(self.reward_history)[-10:] if self.reward_history else [0]
        avg_recent_reward = np.mean(recent_rewards)
        if avg_recent_reward < -0.05:
            perf_level = 0
        elif avg_recent_reward > 0.05:
            perf_level = 2
        else:
            perf_level = 1
            
        return (spread_level, momentum, position_level, perf_level)
    
    def _get_best_price(self, snapshot, side):
        """Récupère le meilleur prix"""
        keys_to_try = {
            'bid': ['best_bid', 'bids', 'bid', 'buy_orders'],
            'ask': ['best_ask', 'asks', 'ask', 'sell_orders']
        }
        
        for key in keys_to_try[side]:
            if key in snapshot:
                data = snapshot[key]
                if isinstance(data, (int, float)):
                    return float(data)
                elif isinstance(data, (list, tuple)) and len(data) > 0:
                    if isinstance(data[0], (int, float)):
                        return float(data[0])
                    elif isinstance(data[0], (list, tuple)) and len(data[0]) >= 2:
                        return float(data[0][0])
        return None
    
    def choose_action(self, state):
        """Epsilon-greedy avec décroissance"""
        if random.random() < self.epsilon:
            self.exploration_count += 1
            return random.randint(0, self.n_actions - 1)
        
        self.exploitation_count += 1
        
        if state not in self.q_table:
            self.q_table[state] = {i: 0.0 for i in range(self.n_actions)}
            return random.randint(0, self.n_actions - 1)
        
        return max(self.q_table[state], key=self.q_table[state].get)
    
    def execute_action(self, action, order_book):
        """Exécute l'action avec gestion des risques"""
        snapshot = order_book.snapshot()
        best_bid = self._get_best_price(snapshot, 'bid')
        best_ask = self._get_best_price(snapshot, 'ask')
        
        if best_bid is None or best_ask is None:
            return []
        
        trades = []
        base_qty = random.randint(30, self.max_order_size)
        
        # Gestion des limites de position
        if action in [1, 3] and self.inventory >= self.position_limit:  # actions d'achat
            return []
        if action in [2, 4] and self.inventory <= -self.position_limit:  # actions de vente
            return []
        
        if action == 0:  # hold
            pass
            
        elif action == 1:  # buy_aggressive (market order)
            trades = order_book.add_order(Order(
                side=Side.BUY, qty=base_qty, order_type=OrderType.MARKET, 
                agent_id=self.agent_id
            ))
            if trades:
                total_qty = sum(t.qty for t in trades)
                avg_price = sum(t.price * t.qty for t in trades) / total_qty
                self.inventory += total_qty
                self.cash -= avg_price * total_qty
                self.trade_count += 1
                
        elif action == 2:  # sell_aggressive (market order)
            trades = order_book.add_order(Order(
                side=Side.SELL, qty=base_qty, order_type=OrderType.MARKET, 
                agent_id=self.agent_id
            ))
            if trades:
                total_qty = sum(t.qty for t in trades)
                avg_price = sum(t.price * t.qty for t in trades) / total_qty
                self.inventory -= total_qty
                self.cash += avg_price * total_qty
                self.trade_count += 1
                
        elif action == 3:  # buy_passive (limit order près du bid)
            limit_price = round(best_bid + random.uniform(0.001, 0.01), 3)
            order_book.add_order(Order(
                side=Side.BUY, qty=base_qty, order_type=OrderType.LIMIT, 
                price=limit_price, agent_id=self.agent_id
            ))
            
        elif action == 4:  # sell_passive (limit order près de l'ask)
            limit_price = round(best_ask - random.uniform(0.001, 0.01), 3)
            order_book.add_order(Order(
                side=Side.SELL, qty=base_qty, order_type=OrderType.LIMIT, 
                price=limit_price, agent_id=self.agent_id
            ))
        
        return trades
    
    def calculate_reward(self, trades, mid_price, action):
        """Fonction de récompense sophistiquée"""
        reward = 0
        
        # Récompense directe des trades
        for trade in trades:
            if trade == Side.BUY:
                # Récompense si on achète en dessous du mid
                price_advantage = mid_price - trade.price
                reward += price_advantage * trade.qty * 10  # amplification
            else:
                # Récompense si on vend au dessus du mid
                price_advantage = trade.price - mid_price
                reward += price_advantage * trade.qty * 10
        
        # Récompense pour la gestion d'inventaire
        inventory_penalty = (abs(self.inventory) / self.position_limit) ** 2 * 0.1
        reward -= inventory_penalty
        
        # Bonus pour les actions passives (fournir de la liquidité)
        if action in [3, 4]:  # ordres limite
            reward += 0.02
        
        # Pénalité pour l'inaction excessive
        recent_actions = [a[1] for a in self.action_history[-5:]]
        if len(recent_actions) >= 5 and all(a == 0 for a in recent_actions):
            reward -= 0.05
        
        # Bonus pour la rentabilité
        if len(self.price_history) >= 2:
            price_change = self.price_history[-1] - self.price_history[-2]
            if self.inventory > 0 and price_change > 0:  # long et prix monte
                reward += self.inventory * price_change * 0.1
            elif self.inventory < 0 and price_change < 0:  # short et prix baisse
                reward += abs(self.inventory) * abs(price_change) * 0.1
        
        return reward
    
    def update_q_table(self, old_state, action, reward, new_state):
        """Mise à jour Q-table avec learning rate adaptatif"""
        if old_state not in self.q_table:
            self.q_table[old_state] = {i: 0.0 for i in range(self.n_actions)}
        if new_state not in self.q_table:
            self.q_table[new_state] = {i: 0.0 for i in range(self.n_actions)}
        
        # Q-learning avec learning rate adaptatif
        old_q = self.q_table[old_state][action]
        max_future_q = max(self.q_table[new_state].values()) if self.q_table[new_state] else 0
        
        # Learning rate qui diminue avec l'expérience sur cet état-action
        visit_count = sum(1 for s, a in self.action_history if s == old_state and a == action)
        adapted_lr = self.learning_rate / (1 + visit_count * 0.01)
        
        new_q = old_q + adapted_lr * (reward + self.discount_factor * max_future_q - old_q)
        self.q_table[old_state][action] = new_q
        
        # Décroissance epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def get_portfolio_value(self, current_mid_price):
        """Calcule la valeur du portefeuille"""
        return self.cash + self.inventory * current_mid_price
    
    def get_stats(self, current_mid_price):
        """Statistiques détaillées"""
        portfolio_value = self.get_portfolio_value(current_mid_price)
        total_pnl = portfolio_value - self.initial_cash
        
        return {
            'inventory': self.inventory,
            'cash': self.cash,
            'portfolio_value': portfolio_value,
            'total_pnl': total_pnl,
            'trade_count': self.trade_count,
            'q_table_size': len(self.q_table),
            'avg_reward': np.mean(list(self.reward_history)[-20:]) if len(self.reward_history) >= 20 else 0,
            'epsilon': self.epsilon,
            'exploration_ratio': self.exploration_count / (self.exploration_count + self.exploitation_count) if (self.exploration_count + self.exploitation_count) > 0 else 0
        }

# Agent naïf pour comparaison
class NaiveAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.inventory = 0
        self.cash = 100000
        self.initial_cash = 100000
        self.trade_count = 0
        self.position_limit = 200
        
    def execute_action(self, order_book):
        """Stratégie simple : acheter bas, vendre haut avec probabilité"""
        if random.random() < 0.2:  # 20% de chance d'action
            snapshot = order_book.snapshot()
            best_bid = self._get_best_price(snapshot, 'bid')
            best_ask = self._get_best_price(snapshot, 'ask')
            
            if best_bid is None or best_ask is None:
                return []
            
            trades = []
            qty = random.randint(30, 100)
            
            # Stratégie simple basée sur l'inventaire
            if self.inventory < -self.position_limit:
                # Trop short, acheter
                trades = order_book.add_order(Order(
                    side=Side.BUY, qty=qty, order_type=OrderType.MARKET, 
                    agent_id=self.agent_id
                ))
            elif self.inventory > self.position_limit:
                # Trop long, vendre
                trades = order_book.add_order(Order(
                    side=Side.SELL, qty=qty, order_type=OrderType.MARKET, 
                    agent_id=self.agent_id
                ))
            else:
                # Action aléatoire
                side = random.choice([Side.BUY, Side.SELL])
                trades = order_book.add_order(Order(
                    side=side, qty=qty, order_type=OrderType.MARKET, 
                    agent_id=self.agent_id
                ))
            
            # Mise à jour de l'état
            if trades:
                for trade in trades:
                    if trade == Side.BUY:
                        self.inventory += trade.qty
                        self.cash -= trade.price * trade.qty
                    else:
                        self.inventory -= trade.qty
                        self.cash += trade.price * trade.qty
                self.trade_count += 1
            
            return trades
        return []
    
    def _get_best_price(self, snapshot, side):
        """Récupère le meilleur prix"""
        keys_to_try = {
            'bid': ['best_bid', 'bids', 'bid', 'buy_orders'],
            'ask': ['best_ask', 'asks', 'ask', 'sell_orders']
        }
        
        for key in keys_to_try[side]:
            if key in snapshot:
                data = snapshot[key]
                if isinstance(data, (int, float)):
                    return float(data)
                elif isinstance(data, (list, tuple)) and len(data) > 0:
                    if isinstance(data[0], (int, float)):
                        return float(data[0])
                    elif isinstance(data[0], (list, tuple)) and len(data[0]) >= 2:
                        return float(data[0][0])
        return None
    
    def get_portfolio_value(self, current_mid_price):
        return self.cash + self.inventory * current_mid_price
    
    def get_stats(self, current_mid_price):
        portfolio_value = self.get_portfolio_value(current_mid_price)
        return {
            'inventory': self.inventory,
            'cash': self.cash,
            'portfolio_value': portfolio_value,
            'total_pnl': portfolio_value - self.initial_cash,
            'trade_count': self.trade_count
        }

# Initialisation
order_book = OrderBook("AAPL")
market_makers = ["MM1", "MM2"]
aggressive_trader = "TA1"
rl_agent = ImprovedRLAgent("RL_AGENT")
naive_agent = NaiveAgent("NAIVE_AGENT")

# Historiques
price_history = deque(maxlen=200)
bid_history = deque(maxlen=200)
ask_history = deque(maxlen=200)
spread_history = deque(maxlen=200)

# Historiques des agents
rl_pnl_history = deque(maxlen=200)
naive_pnl_history = deque(maxlen=200)
rl_inventory_history = deque(maxlen=200)
naive_inventory_history = deque(maxlen=200)
rl_reward_history = deque(maxlen=200)

# Market maker et aggressive trader (inchangés)
def market_maker_action(agent_id):
    mid = order_book.mid_price()
    if mid is None:
        mid = 100.0
    spread = random.uniform(0.05, 0.2)
    qty = random.randint(50, 150)

    bid_price = round(mid - spread/2, 3)
    ask_price = round(mid + spread/2, 3)

    order_book.add_order(Order(side=Side.BUY, qty=qty, order_type=OrderType.LIMIT, price=bid_price, agent_id=agent_id))
    order_book.add_order(Order(side=Side.SELL, qty=qty, order_type=OrderType.LIMIT, price=ask_price, agent_id=agent_id))

def aggressive_trader_action(agent_id):
    if random.random() < 0.15:  # 15% de chance
        side = random.choice([Side.BUY, Side.SELL])
        qty = random.randint(50, 200)
        trades = order_book.add_order(Order(side=side, qty=qty, order_type=OrderType.MARKET, agent_id=agent_id))
        for t in trades:
            price_history.append(t.price)

# Visualisation
plt.ion()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

print("Simulation comparative : Agent RL vs Agent Naïf")
print("-" * 90)
print(f"{'Tick':<4} {'Best Bid':<9} {'Best Ask':<9} {'RL Action':<12} {'RL PnL':<8} {'Naive PnL':<10} {'RL ε':<6}")
print("-" * 90)

n_ticks = 300  # Plus de ticks pour l'apprentissage

for tick in range(n_ticks):
    # Market makers et aggressive trader
    for mm in market_makers:
        market_maker_action(mm)
    aggressive_trader_action(aggressive_trader)
    
    # Récupération des prix
    snapshot = order_book.snapshot()
    best_bid = rl_agent._get_best_price(snapshot, 'bid')
    best_ask = rl_agent._get_best_price(snapshot, 'ask')
    
    if best_bid is not None and best_ask is not None:
        current_mid_price = (best_bid + best_ask) / 2
        spread = round(best_ask - best_bid, 4)
        bid_history.append(best_bid)
        ask_history.append(best_ask)
        spread_history.append(spread)
    else:
        current_mid_price = 100.0
        bid_history.append(None)
        ask_history.append(None)
        spread_history.append(None)
    
    # Agent RL
    current_state = rl_agent.get_state(order_book, tick)
    action = rl_agent.choose_action(current_state)
    rl_trades = rl_agent.execute_action(action, order_book)
    
    # Ajouter les trades RL aux prix
    for t in rl_trades:
        price_history.append(t.price)
    
    # Récompense et apprentissage RL
    reward = rl_agent.calculate_reward(rl_trades, current_mid_price, action)
    rl_agent.reward_history.append(reward)
    rl_reward_history.append(reward)
    
    if rl_agent.last_state is not None:
        rl_agent.update_q_table(rl_agent.last_state, rl_agent.last_action, reward, current_state)
    
    rl_agent.last_state = current_state
    rl_agent.last_action = action
    rl_agent.action_history.append((current_state, action))
    
    # Agent naïf
    naive_trades = naive_agent.execute_action(order_book)
    for t in naive_trades:
        price_history.append(t.price)
    
    # Mise à jour des historiques de performance
    rl_stats = rl_agent.get_stats(current_mid_price)
    naive_stats = naive_agent.get_stats(current_mid_price)
    
    rl_pnl_history.append(rl_stats['total_pnl'])
    naive_pnl_history.append(naive_stats['total_pnl'])
    rl_inventory_history.append(rl_stats['inventory'])
    naive_inventory_history.append(naive_stats['inventory'])
    
    # Affichage
    bid_str = f"{best_bid:.3f}" if best_bid is not None else "N/A"
    ask_str = f"{best_ask:.3f}" if best_ask is not None else "N/A"
    action_str = rl_agent.actions[action]
    
    print(f"{tick+1:<4} {bid_str:<9} {ask_str:<9} {action_str:<12} {rl_stats['total_pnl']:<8.1f} {naive_stats['total_pnl']:<10.1f} {rl_stats['epsilon']:<6.3f}")
    
    # Graphiques (tous les 10 ticks)
    if tick % 10 == 0 and tick > 0:
        # Graphique 1: Comparaison des PnL
        ax1.clear()
        ax1.plot(list(rl_pnl_history), 'blue', label='RL Agent', linewidth=2)
        ax1.plot(list(naive_pnl_history), 'red', label='Naive Agent', linewidth=2)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_title("PnL Comparison")
        ax1.set_xlabel("Tick")
        ax1.set_ylabel("PnL")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Inventaires
        ax2.clear()
        ax2.plot(list(rl_inventory_history), 'blue', label='RL Inventory', linewidth=2)
        ax2.plot(list(naive_inventory_history), 'red', label='Naive Inventory', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title("Inventory Comparison")
        ax2.set_xlabel("Tick")
        ax2.set_ylabel("Position")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Graphique 3: Prix et spread
        ax3.clear()
        if len(bid_history) > 0:
            x_bids = [i for i, bid in enumerate(bid_history) if bid is not None]
            y_bids = [bid for bid in bid_history if bid is not None]
            x_asks = [i for i, ask in enumerate(ask_history) if ask is not None]
            y_asks = [ask for ask in ask_history if ask is not None]
            
            if x_bids:
                ax3.plot(x_bids, y_bids, 'g-', label='Best Bid', linewidth=1, alpha=0.7)
            if x_asks:
                ax3.plot(x_asks, y_asks, 'r-', label='Best Ask', linewidth=1, alpha=0.7)
        
        ax3.set_title("Market Prices")
        ax3.set_xlabel("Tick")
        ax3.set_ylabel("Price")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Graphique 4: Récompenses RL et métriques
        ax4.clear()
        if rl_reward_history:
            ax4.plot(list(rl_reward_history), 'orange', alpha=0.6, linewidth=1)
            if len(rl_reward_history) >= 20:
                # Moyenne mobile
                window = 20
                moving_avg = []
                for i in range(window-1, len(rl_reward_history)):
                    avg = sum(list(rl_reward_history)[i-window+1:i+1]) / window
                    moving_avg.append(avg)
                ax4.plot(range(window-1, len(rl_reward_history)), moving_avg, 'darkblue', linewidth=2, label=f'MA({window})')
                ax4.legend()
        
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title("RL Learning Progress (Rewards)")
        ax4.set_xlabel("Tick")
        ax4.set_ylabel("Reward")
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.05)

print("-" * 90)
print("\n=== RÉSULTATS FINAUX ===")
final_rl_stats = rl_agent.get_stats(current_mid_price)
final_naive_stats = naive_agent.get_stats(current_mid_price)

print(f"\n🤖 AGENT RL:")
for key, value in final_rl_stats.items():
    print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

print(f"\n🎯 AGENT NAÏF:")
for key, value in final_naive_stats.items():
    print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

print(f"\n📊 COMPARAISON:")
print(f"  Avantage PnL RL: {final_rl_stats['total_pnl'] - final_naive_stats['total_pnl']:.2f}")
print(f"  Ratio trades RL/Naïf: {final_rl_stats['trade_count']/max(final_naive_stats['trade_count'],1):.2f}")
print(f"  États appris: {final_rl_stats['q_table_size']}")

plt.ioff()
plt.show()