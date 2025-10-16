import random
import time
import matplotlib.pyplot as plt
from collections import deque
from OrderBook import OrderBook, Order, Side, OrderType

# Initialisation
order_book = OrderBook("AAPL")
market_makers = ["MM1", "MM2"]
aggressive_trader = "TA1"

# Historique des données
price_history = deque(maxlen=50)  # garde les 50 derniers prix
bid_history = deque(maxlen=50)    # historique des meilleurs bid
ask_history = deque(maxlen=50)    # historique des meilleurs ask
spread_history = deque(maxlen=50) # historique du spread

# --- Agents ---
def market_maker_action(agent_id):
    mid = order_book.mid_price()
    if mid is None:
        mid = 100.0
    spread = random.uniform(0.1, 0.3)  # spread variable
    qty = random.randint(50, 150)

    bid_price = round(mid - spread/2, 2)
    ask_price = round(mid + spread/2, 2)

    order_book.add_order(Order(side=Side.BUY, qty=qty, order_type=OrderType.LIMIT, price=bid_price, agent_id=agent_id))
    order_book.add_order(Order(side=Side.SELL, qty=qty, order_type=OrderType.LIMIT, price=ask_price, agent_id=agent_id))

def aggressive_trader_action(agent_id):
    if random.random() < 0.7:  # 30% de chance d'action
        side = random.choice([Side.BUY, Side.SELL])
        qty = random.randint(50, 200)
        trades = order_book.add_order(Order(side=side, qty=qty, order_type=OrderType.MARKET, agent_id=agent_id))
        for t in trades:
            price_history.append(t.price)






# --- Visualisation ---
plt.ion()
fig, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize = (14, 4))


print("Début de la simulation - Affichage Bid/Ask par tick")
print("-" * 60)
print(f"{'Tick':<6} {'Best Bid':<10} {'Best Ask':<10} {'Spread':<8} {'Mid Price':<10}")
print("-" * 60)

for tick in range(50):  # 50 ticks
    # Agents agissent
    for mm in market_makers:
        market_maker_action(mm)
    aggressive_trader_action(aggressive_trader)

    # Debug : affichons d'abord le contenu du snapshot
    snapshot = order_book.snapshot()
    #print(f"Debug Tick {tick+1}: snapshot keys = {list(snapshot.keys()) if isinstance(snapshot, dict) else 'Not a dict'}")
    
    # Essayons différentes clés possibles pour récupérer les données
    bids = None
    asks = None
    
    # Essayons plusieurs clés possibles
    for bid_key in ["best_bid"]:
        if bid_key in snapshot:
            bids = snapshot[bid_key]
            break
    
    for ask_key in ["best_ask"]:
        if ask_key in snapshot:
            asks = snapshot[ask_key]
            break
    
    # Récupération des meilleurs prix
    best_bid = None
    best_ask = None
    
    # Pour les bids
    if bids is not None:
        if isinstance(bids, (int, float)):
            best_bid = float(bids)
        elif isinstance(bids, (list, tuple)) and len(bids) > 0:
            if isinstance(bids[0], (int, float)):
                best_bid = float(bids[0])
            elif isinstance(bids[0], (list, tuple)) and len(bids[0]) >= 2:
                best_bid = float(bids[0][0])
    
    # Pour les asks
    if asks is not None:
        if isinstance(asks, (int, float)):
            best_ask = float(asks)
        elif isinstance(asks, (list, tuple)) and len(asks) > 0:
            if isinstance(asks[0], (int, float)):
                best_ask = float(asks[0])
            elif isinstance(asks[0], (list, tuple)) and len(asks[0]) >= 2:
                best_ask = float(asks[0][0])
    
    # Si on n'a pas trouvé avec snapshot, essayons directement avec OrderBook
    if best_bid is None or best_ask is None:
        try:
            # Essayons d'accéder directement aux attributs de l'OrderBook
            if hasattr(order_book, 'best_bid') and order_book.best_bid() is not None:
                best_bid = float(order_book.best_bid())
            if hasattr(order_book, 'best_ask') and order_book.best_ask() is not None:
                best_ask = float(order_book.best_ask())
        except:
            pass
    
    # Calcul du spread et mid price
    spread = None
    mid_price = order_book.mid_price()
    
    if best_bid is not None and best_ask is not None:
        spread = round(best_ask - best_bid, 3)
        bid_history.append(best_bid)
        ask_history.append(best_ask)
        spread_history.append(spread)
    else:
        bid_history.append(None)
        ask_history.append(None)
        spread_history.append(None)
    
    # Affichage console
    bid_str = f"{best_bid:.2f}" if best_bid is not None else "N/A"
    ask_str = f"{best_ask:.2f}" if best_ask is not None else "N/A"
    spread_str = f"{spread:.3f}" if spread is not None else "N/A"
    mid_str = f"{mid_price:.2f}" if mid_price is not None else "N/A"
    
    print(f"{tick+1:<6} {bid_str:<10} {ask_str:<10} {spread_str:<8} {mid_str:<10}")

    # --- Graphique 1 : profondeur du carnet ---
    # ax1.clear()
    
    # # Affichage simplifié avec les meilleurs bid/ask uniquement
    # if best_bid is not None:
    #     ax1.bar(best_bid, 100, width=0.02, color='green', alpha=0.7, label=f'Best Bid: {best_bid:.2f}')
    # if best_ask is not None:
    #     ax1.bar(best_ask, 100, width=0.02, color='red', alpha=0.7, label=f'Best Ask: {best_ask:.2f}')
    
    # # Si on a les données complètes, on les affiche aussi
    # if isinstance(bids, list) and all(isinstance(b, (list, tuple)) and len(b) >= 2 for b in bids):
    #     bid_prices, bid_qtys = zip(*[(b[0], b[1]) for b in bids])
    #     ax1.bar(bid_prices, bid_qtys, width=0.01, color='green', alpha=0.5, label='All Bids')
    
    # if isinstance(asks, list) and all(isinstance(a, (list, tuple)) and len(a) >= 2 for a in asks):
    #     ask_prices, ask_qtys = zip(*[(a[0], a[1]) for a in asks])
    #     ax1.bar(ask_prices, ask_qtys, width=0.01, color='red', alpha=0.5, label='All Asks')

    # ax1.set_title(f"Order Book Depth (Tick {tick+1})")
    # ax1.set_xlabel("Price")
    # ax1.set_ylabel("Quantity")
    # ax1.legend()
    # ax1.grid(True, alpha=0.3)

    # --- Graphique 2 : historique des prix tradés ---
    ax2.clear()
    if price_history:
        ax2.plot(list(price_history), marker="o", color="black", linewidth=2, markersize=4)
        ax2.set_title("Last Traded Prices")
        ax2.set_xlabel("Tick")
        ax2.set_ylabel("Price")
        ax2.grid(True, alpha=0.3)

    # --- Graphique 3 : évolution bid/ask ---
    ax3.clear()
    
    if len(bid_history) > 0:
        # Créer des listes pour les points valides
        x_bids = []
        y_bids = []
        for i, bid in enumerate(bid_history):
            if bid is not None:
                x_bids.append(i)
                y_bids.append(bid)
        
        if len(x_bids) > 0:
            ax3.plot(x_bids, y_bids, 'g-o', label='Best Bid', linewidth=2, markersize=4)
    
    if len(ask_history) > 0:
        # Créer des listes pour les points valides
        x_asks = []
        y_asks = []
        for i, ask in enumerate(ask_history):
            if ask is not None:
                x_asks.append(i)
                y_asks.append(ask)
        
        if len(x_asks) > 0:
            ax3.plot(x_asks, y_asks, 'r-o', label='Best Ask', linewidth=2, markersize=4)
    
    ax3.set_title("Best Bid/Ask Evolution")
    ax3.set_xlabel("Tick")
    ax3.set_ylabel("Price")
    if len(bid_history) > 0 or len(ask_history) > 0:
        ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- Graphique 4 : évolution du spread ---
    ax4.clear()
    
    if len(spread_history) > 0:
        # Créer des listes pour les points valides
        x_spreads = []
        y_spreads = []
        for i, spread_val in enumerate(spread_history):
            if spread_val is not None and spread_val > 0:
                x_spreads.append(i)
                y_spreads.append(spread_val)
        
        if len(x_spreads) > 0:
            ax4.plot(x_spreads, y_spreads, 'b-o', linewidth=2, markersize=4)
            ax4.set_title("Bid-Ask Spread Evolution")
            ax4.set_xlabel("Tick")
            ax4.set_ylabel("Spread")
        else:
            ax4.text(0.5, 0.5, 'No valid spread data', transform=ax4.transAxes, ha='center')
    
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.pause(0.5)

print("-" * 60)
print("Simulation terminée")

plt.ioff()
plt.show()