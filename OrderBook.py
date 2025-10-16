# Fix syntax error and rerun the module/demo creation
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque, defaultdict
from typing import Deque, Dict, List, Optional, Tuple, Iterable
import heapq
import time
import math
import json
import random

# -----------------------------
# Order Book Implementation
# -----------------------------

class Side(Enum):
    BUY = auto()
    SELL = auto()

class OrderType(Enum):
    LIMIT = auto()
    MARKET = auto()

@dataclass
class Order:
    side: Side
    qty: float
    order_type: OrderType
    price: Optional[float] = None
    agent_id: Optional[str] = None
    order_id: Optional[str] = None
    ts: float = field(default_factory=lambda: time.time())

    def __post_init__(self):
        if self.order_type == OrderType.LIMIT and (self.price is None):
            raise ValueError("Limit order requires a price")
        if self.order_type == OrderType.MARKET:
            self.price = None

@dataclass
class Trade:
    price: float
    qty: float
    buy_order_id: str
    sell_order_id: str
    ts: float

class PriceLevelQueue:
    """FIFO queue for orders at the same price level to enforce time priority (price-time priority)."""
    def __init__(self):
        self.q: Deque[Order] = deque()
        self.total_qty: float = 0.0

    def append(self, order: Order):
        self.q.append(order)
        self.total_qty += order.qty

    def popleft(self) -> Order:
        o = self.q.popleft()
        self.total_qty -= o.qty
        return o

    def __len__(self):
        return len(self.q)

    def peek(self) -> Optional[Order]:
        return self.q[0] if self.q else None

class OrderBook:
    """
    Order book with price-time priority, supporting:
      - Limit and Market orders
      - Matching engine (continuous, immediate upon insert)
      - Cancel by order_id
      - Depth snapshot, BBO, mid, last price
      - VWAP/TWAP over rolling windows
    Data structures:
      - For each side, maintain {price -> FIFO queue of orders} (time priority)
      - Maintain a heap of prices to quickly find best price
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        # Map price -> FIFO queue
        self.bids: Dict[float, PriceLevelQueue] = defaultdict(PriceLevelQueue)
        self.asks: Dict[float, PriceLevelQueue] = defaultdict(PriceLevelQueue)
        # Heaps of prices
        self._bid_heap: List[float] = []  # store negative prices for max-heap behavior
        self._ask_heap: List[float] = []  # min-heap
        # Order lookup for cancellation
        self._order_index: Dict[str, Tuple[Side, float]] = {}
        # Trades
        self.trades: List[Trade] = []
        self.last_traded_price: Optional[float] = None

    # ---------- Utilities ----------
    def _push_price(self, side: Side, price: float):
        if side == Side.BUY:
            heapq.heappush(self._bid_heap, -price)
        else:
            heapq.heappush(self._ask_heap, price)

    def _best_price(self, side: Side) -> Optional[float]:
        if side == Side.BUY:
            # Clean stale prices
            while self._bid_heap:
                p = -self._bid_heap[0]
                if p in self.bids and len(self.bids[p]) > 0:
                    break
                heapq.heappop(self._bid_heap)
            return -self._bid_heap[0] if self._bid_heap else None
        else:
            while self._ask_heap:
                p = self._ask_heap[0]
                if p in self.asks and len(self.asks[p]) > 0:
                    break
                heapq.heappop(self._ask_heap)
            return self._ask_heap[0] if self._ask_heap else None

    def best_bid(self) -> Optional[float]:
        return self._best_price(Side.BUY)

    def best_ask(self) -> Optional[float]:
        return self._best_price(Side.SELL)

    def mid_price(self) -> Optional[float]:
        b, a = self.best_bid(), self.best_ask()
        if b is not None and a is not None:
            return 0.5 * (b + a)
        return None

    def spot_price(self, side: Optional[Side] = None) -> Optional[float]:
        """Spot as best executable: if side is BUY -> best ask; if SELL -> best bid; else mid if both exist; else last trade."""
        if side == Side.BUY:
            return self.best_ask()
        if side == Side.SELL:
            return self.best_bid()
        m = self.mid_price()
        return m if m is not None else self.last_traded_price

    # ---------- Public API ----------
    def add_order(self, order: Order) -> List[Trade]:
        """Add order (limit or market). Return list of trades executed."""
        if order.order_id is None:
            order.order_id = f"{order.agent_id or 'anon'}-{int(order.ts*1e6)}-{random.randint(1000,9999)}"
        trades: List[Trade] = []
        if order.order_type == OrderType.MARKET:
            trades = self._execute_market(order)
        else:
            trades = self._add_limit(order)
        return trades

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing (unfilled) order by ID. Returns True if found & removed."""
        info = self._order_index.pop(order_id, None)
        if not info:
            return False
        side, price = info
        book = self.bids if side == Side.BUY else self.asks
        q = book.get(price)
        if not q:
            return False
        # Remove from queue by scanning (rare)
        removed = False
        new_q = deque()
        while q.q:
            o = q.q.popleft()
            if o.order_id == order_id and not removed:
                removed = True
                q.total_qty -= o.qty
            else:
                new_q.append(o)
        q.q = new_q
        if len(q) == 0:
            # remove empty level
            book.pop(price, None)
        return removed

    def depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        """Return top N levels: {'bids': [(price, qty), ...], 'asks': [(price, qty), ...]}"""
        bid_levels = []
        ask_levels = []
        # Collect bids (desc)
        tmp = []
        for p in self.bids.keys():
            if len(self.bids[p]) > 0:
                tmp.append(p)
        tmp.sort(reverse=True)
        for p in tmp[:levels]:
            bid_levels.append((p, self.bids[p].total_qty))
        # Collect asks (asc)
        tmp = []
        for p in self.asks.keys():
            if len(self.asks[p]) > 0:
                tmp.append(p)
        tmp.sort()
        for p in tmp[:levels]:
            ask_levels.append((p, self.asks[p].total_qty))
        return {"bids": bid_levels, "asks": ask_levels}

    def vwap(self, window_seconds: Optional[float] = None, last_n_trades: Optional[int] = None) -> Optional[float]:
        """Compute VWAP over a time window (seconds) or last N trades."""
        trades = self.trades
        if not trades:
            return None
        if window_seconds is not None:
            t_cut = time.time() - window_seconds
            trades = [t for t in trades if t.ts >= t_cut]
        if last_n_trades is not None and last_n_trades < len(trades):
            trades = trades[-last_n_trades:]
        vol = sum(t.qty for t in trades)
        if vol == 0:
            return None
        return sum(t.price * t.qty for t in trades) / vol

    def twap(self, window_seconds: float = 60.0) -> Optional[float]:
        """Compute TWAP over a recent window using last trade prices as proxy samples."""
        if not self.trades:
            return None
        t_cut = time.time() - window_seconds
        prices = [t.price for t in self.trades if t.ts >= t_cut]
        if not prices:
            prices = [self.trades[-1].price]
        return sum(prices) / len(prices)

    def snapshot(self) -> Dict:
        """State for publication (e.g., via MCP): BBO, mid, last, depth."""
        b, a = self.best_bid(), self.best_ask()
        snap = {
            "symbol": self.symbol,
            "best_bid": b,
            "best_ask": a,
            "mid": self.mid_price(),
            "last": self.last_traded_price,
            "depth": self.depth(5),
            "ts": time.time(),
        }
        return snap

    # ---------- Internal Mechanics ----------
    def _add_limit(self, order: Order) -> List[Trade]:
        # Match aggressively if crossing, otherwise rest in book
        trades: List[Trade] = []
        if order.side == Side.BUY:
            # Match against asks priced <= order.price
            while order.qty > 1e-12:
                best_ask = self.best_ask()
                if best_ask is None or best_ask > order.price:
                    break
                q = self.asks[best_ask]
                head = q.peek()
                fill_qty = min(order.qty, head.qty)
                trades.append(self._execute_trade(price=best_ask, qty=fill_qty,
                                                  buy_id=order.order_id, sell_id=head.order_id))
                order.qty -= fill_qty
                head.qty -= fill_qty
                q.total_qty -= fill_qty
                if head.qty <= 1e-12:
                    q.popleft()
                    if len(q) == 0:
                        self.asks.pop(best_ask, None)
            # Rest unfilled
            if order.qty > 1e-12:
                price = order.price
                if price not in self.bids or len(self.bids[price]) == 0:
                    self._push_price(Side.BUY, price)
                self.bids[price].append(order)
                self._order_index[order.order_id] = (Side.BUY, price)
        else:
            # SELL: match against bids priced >= order.price
            while order.qty > 1e-12:
                best_bid = self.best_bid()
                if best_bid is None or best_bid < order.price:
                    break
                q = self.bids[best_bid]
                head = q.peek()
                fill_qty = min(order.qty, head.qty)
                trades.append(self._execute_trade(price=best_bid, qty=fill_qty,
                                                  buy_id=head.order_id, sell_id=order.order_id))
                order.qty -= fill_qty
                head.qty -= fill_qty
                q.total_qty -= fill_qty
                if head.qty <= 1e-12:
                    q.popleft()
                    if len(q) == 0:
                        self.bids.pop(best_bid, None)
            if order.qty > 1e-12:
                price = order.price
                if price not in self.asks or len(self.asks[price]) == 0:
                    self._push_price(Side.SELL, price)
                self.asks[price].append(order)
                self._order_index[order.order_id] = (Side.SELL, price)
        return trades

    def _execute_market(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        if order.side == Side.BUY:
            while order.qty > 1e-12:
                best_ask = self.best_ask()
                if best_ask is None:
                    break
                q = self.asks[best_ask]
                head = q.peek()
                fill_qty = min(order.qty, head.qty)
                trades.append(self._execute_trade(price=best_ask, qty=fill_qty,
                                                  buy_id=order.order_id, sell_id=head.order_id))
                order.qty -= fill_qty
                head.qty -= fill_qty
                q.total_qty -= fill_qty
                if head.qty <= 1e-12:
                    q.popleft()
                    if len(q) == 0:
                        self.asks.pop(best_ask, None)
        else:
            while order.qty > 1e-12:
                best_bid = self.best_bid()
                if best_bid is None:
                    break
                q = self.bids[best_bid]
                head = q.peek()
                fill_qty = min(order.qty, head.qty)
                trades.append(self._execute_trade(price=best_bid, qty=fill_qty,
                                                  buy_id=head.order_id, sell_id=order.order_id))
                order.qty -= fill_qty
                head.qty -= fill_qty
                q.total_qty -= fill_qty
                if head.qty <= 1e-12:
                    q.popleft()
                    if len(q) == 0:
                        self.bids.pop(best_bid, None)
        return trades

    def _execute_trade(self, price: float, qty: float, buy_id: str, sell_id: str) -> Trade:
        t = Trade(price=price, qty=qty, buy_order_id=buy_id, sell_order_id=sell_id, ts=time.time())
        self.trades.append(t)
        self.last_traded_price = price
        return t

# -----------------------------
# Minimal MCP-like publisher (stub)
# -----------------------------
class MCPLikePublisher:
    """Stub that would publish snapshots/events to agents. Here we just buffer JSON events."""
    def __init__(self):
        self.events: List[str] = []

    def publish_snapshot(self, snap: Dict):
        self.events.append(json.dumps({"type": "snapshot", "data": snap}))

    def publish_trade(self, trade: Trade):
        self.events.append(json.dumps({
            "type": "trade",
            "data": {"price": trade.price, "qty": trade.qty, "ts": trade.ts,
                     "buy_id": trade.buy_order_id, "sell_id": trade.sell_order_id}
        }))

# -----------------------------
# Demo / Quick Test
# -----------------------------

def demo():
    ob = OrderBook("AAPL")
    bus = MCPLikePublisher()

    # Seed book with some resting liquidity
    for px, qty in [(99.8, 500), (99.9, 400), (100.0, 300)]:
        ob.add_order(Order(side=Side.BUY, qty=qty, order_type=OrderType.LIMIT, price=px, agent_id="MM1"))
    for px, qty in [(100.2, 500), (100.1, 400), (100.0, 200)]:
        ob.add_order(Order(side=Side.SELL, qty=qty, order_type=OrderType.LIMIT, price=px, agent_id="MM2"))

    bus.publish_snapshot(ob.snapshot())

    # Cross the spread with a BUY market order
    trades = ob.add_order(Order(side=Side.BUY, qty=350, order_type=OrderType.MARKET, agent_id="AGENT_MKT"))
    for t in trades:
        bus.publish_trade(t)

    # Post a crossing SELL limit at 99.9 (should match immediately against best bid 100.0)
    trades2 = ob.add_order(Order(side=Side.SELL, qty=250, order_type=OrderType.LIMIT, price=99.9, agent_id="AGENT_CROSS"))
    for t in trades2:
        bus.publish_trade(t)

    # Cancel a remaining order (add then cancel to demonstrate)
    tmp_order = Order(side=Side.BUY, qty=150, order_type=OrderType.LIMIT, price=99.7, agent_id="TEMP")
    ob.add_order(tmp_order)
    ob.cancel_order(tmp_order.order_id)

    # Publish a new snapshot
    bus.publish_snapshot(ob.snapshot())

    # Compute market metrics
    metrics = {
        "best_bid": ob.best_bid(),
        "best_ask": ob.best_ask(),
        "mid": ob.mid_price(),
        "last": ob.last_traded_price,
        "depth_top3": ob.depth(3),
        "vwap_last_5_trades": ob.vwap(last_n_trades=5),
        "twap_60s": ob.twap(window_seconds=60.0),
        "spot_buy": ob.spot_price(Side.BUY),
        "spot_sell": ob.spot_price(Side.SELL),
    }

    return ob, bus, metrics

ob, bus, metrics = demo()

# Save module and demo artifacts
module_src = """
# Auto-generated Order Book module (copy-paste into your project as order_book.py)
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque, defaultdict
from typing import Deque, Dict, List, Optional, Tuple
import heapq, time, json, random

class Side(Enum):
    BUY = auto()
    SELL = auto()

class OrderType(Enum):
    LIMIT = auto()
    MARKET = auto()

@dataclass
class Order:
    side: Side
    qty: float
    order_type: OrderType
    price: Optional[float] = None
    agent_id: Optional[str] = None
    order_id: Optional[str] = None
    ts: float = field(default_factory=lambda: time.time())

    def __post_init__(self):
        if self.order_type == OrderType.LIMIT and (self.price is None):
            raise ValueError("Limit order requires a price")
        if self.order_type == OrderType.MARKET:
            self.price = None

@dataclass
class Trade:
    price: float
    qty: float
    buy_order_id: str
    sell_order_id: str
    ts: float

class PriceLevelQueue:
    def __init__(self):
        self.q: Deque[Order] = deque()
        self.total_qty: float = 0.0
    def append(self, order: Order):
        self.q.append(order); self.total_qty += order.qty
    def popleft(self) -> Order:
        o = self.q.popleft(); self.total_qty -= o.qty; return o
    def __len__(self): return len(self.q)
    def peek(self) -> Optional[Order]: return self.q[0] if self.q else None

class OrderBook:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: Dict[float, PriceLevelQueue] = defaultdict(PriceLevelQueue)
        self.asks: Dict[float, PriceLevelQueue] = defaultdict(PriceLevelQueue)
        self._bid_heap: List[float] = []
        self._ask_heap: List[float] = []
        self._order_index: Dict[str, Tuple[Side, float]] = {}
        self.trades: List[Trade] = []
        self.last_traded_price: Optional[float] = None

    def _push_price(self, side: Side, price: float):
        import heapq
        if side == Side.BUY: heapq.heappush(self._bid_heap, -price)
        else: heapq.heappush(self._ask_heap, price)

    def _best_price(self, side: Side) -> Optional[float]:
        import heapq
        if side == Side.BUY:
            while self._bid_heap:
                p = -self._bid_heap[0]
                if p in self.bids and len(self.bids[p]) > 0: break
                heapq.heappop(self._bid_heap)
            return -self._bid_heap[0] if self._bid_heap else None
        else:
            while self._ask_heap:
                p = self._ask_heap[0]
                if p in self.asks and len(self.asks[p]) > 0: break
                heapq.heappop(self._ask_heap)
            return self._ask_heap[0] if self._ask_heap else None

    def best_bid(self): return self._best_price(Side.BUY)
    def best_ask(self): return self._best_price(Side.SELL)

    def mid_price(self) -> Optional[float]:
        b, a = self.best_bid(), self.best_ask()
        return 0.5*(b+a) if (b is not None and a is not None) else None

    def spot_price(self, side: Optional[Side] = None) -> Optional[float]:
        if side == Side.BUY: return self.best_ask()
        if side == Side.SELL: return self.best_bid()
        m = self.mid_price(); return m if m is not None else self.last_traded_price

    def add_order(self, order: Order) -> List[Trade]:
        import random
        if order.order_id is None:
            order.order_id = f"{order.agent_id or 'anon'}-{int(order.ts*1e6)}-{random.randint(1000,9999)}"
        if order.order_type == OrderType.MARKET: return self._execute_market(order)
        else: return self._add_limit(order)

    def cancel_order(self, order_id: str) -> bool:
        from collections import deque
        info = self._order_index.pop(order_id, None)
        if not info: return False
        side, price = info
        book = self.bids if side == Side.BUY else self.asks
        q = book.get(price)
        if not q: return False
        removed = False; new_q = deque()
        while q.q:
            o = q.q.popleft()
            if o.order_id == order_id and not removed:
                removed = True; q.total_qty -= o.qty
            else: new_q.append(o)
        q.q = new_q
        if len(q) == 0: book.pop(price, None)
        return removed

    def depth(self, levels: int = 5):
        bid_levels, ask_levels = [], []
        tmp = [p for p in self.bids.keys() if len(self.bids[p])>0]; tmp.sort(reverse=True)
        for p in tmp[:levels]: bid_levels.append((p, self.bids[p].total_qty))
        tmp = [p for p in self.asks.keys() if len(self.asks[p])>0]; tmp.sort()
        for p in tmp[:levels]: ask_levels.append((p, self.asks[p].total_qty))
        return {"bids": bid_levels, "asks": ask_levels}

    def vwap(self, window_seconds: Optional[float] = None, last_n_trades: Optional[int] = None) -> Optional[float]:
        import time
        trades = self.trades
        if not trades: return None
        if window_seconds is not None:
            t_cut = time.time() - window_seconds
            trades = [t for t in trades if t.ts >= t_cut]
        if last_n_trades is not None and last_n_trades < len(trades):
            trades = trades[-last_n_trades:]
        vol = sum(t.qty for t in trades)
        if vol == 0: return None
        return sum(t.price*t.qty for t in trades)/vol

    def twap(self, window_seconds: float = 60.0) -> Optional[float]:
        import time
        if not self.trades: return None
        t_cut = time.time() - window_seconds
        prices = [t.price for t in self.trades if t.ts >= t_cut]
        if not prices: prices = [self.trades[-1].price]
        return sum(prices)/len(prices)

    def snapshot(self) -> Dict:
        import time
        return {"symbol": self.symbol, "best_bid": self.best_bid(), "best_ask": self.best_ask(),
                "mid": self.mid_price(), "last": self.last_traded_price, "depth": self.depth(5), "ts": time.time()}

    def _add_limit(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        if order.side == Side.BUY:
            while order.qty > 1e-12:
                best_ask = self.best_ask()
                if best_ask is None or best_ask > order.price: break
                q = self.asks[best_ask]; head = q.peek()
                fill_qty = min(order.qty, head.qty)
                trades.append(self._execute_trade(best_ask, fill_qty, order.order_id, head.order_id))
                order.qty -= fill_qty; head.qty -= fill_qty; q.total_qty -= fill_qty
                if head.qty <= 1e-12:
                    q.popleft(); 
                    if len(q) == 0: self.asks.pop(best_ask, None)
            if order.qty > 1e-12:
                price = order.price
                if price not in self.bids or len(self.bids[price])==0: self._push_price(Side.BUY, price)
                self.bids[price].append(order); self._order_index[order.order_id]=(Side.BUY, price)
        else:
            while order.qty > 1e-12:
                best_bid = self.best_bid()
                if best_bid is None or best_bid < order.price: break
                q = self.bids[best_bid]; head = q.peek()
                fill_qty = min(order.qty, head.qty)
                trades.append(self._execute_trade(best_bid, fill_qty, head.order_id, order.order_id))
                order.qty -= fill_qty; head.qty -= fill_qty; q.total_qty -= fill_qty
                if head.qty <= 1e-12:
                    q.popleft(); 
                    if len(q) == 0: self.bids.pop(best_bid, None)
            if order.qty > 1e-12:
                price = order.price
                if price not in self.asks or len(self.asks[price])==0: self._push_price(Side.SELL, price)
                self.asks[price].append(order); self._order_index[order.order_id]=(Side.SELL, price)
        return trades

    def _execute_market(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        if order.side == Side.BUY:
            while order.qty > 1e-12:
                best_ask = self.best_ask()
                if best_ask is None: break
                q = self.asks[best_ask]; head = q.peek()
                fill_qty = min(order.qty, head.qty)
                trades.append(self._execute_trade(best_ask, fill_qty, order.order_id, head.order_id))
                order.qty -= fill_qty; head.qty -= fill_qty; q.total_qty -= fill_qty
                if head.qty <= 1e-12:
                    q.popleft(); 
                    if len(q) == 0: self.asks.pop(best_ask, None)
        else:
            while order.qty > 1e-12:
                best_bid = self.best_bid()
                if best_bid is None: break
                q = self.bids[best_bid]; head = q.peek()
                fill_qty = min(order.qty, head.qty)
                trades.append(self._execute_trade(best_bid, fill_qty, head.order_id, order.order_id))
                order.qty -= fill_qty; head.qty -= fill_qty; q.total_qty -= fill_qty
                if head.qty <= 1e-12:
                    q.popleft(); 
                    if len(q) == 0: self.bids.pop(best_bid, None)
        return trades

    def _execute_trade(self, price: float, qty: float, buy_id: str, sell_id: str) -> Trade:
        import time
        t = Trade(price=price, qty=qty, buy_order_id=buy_id, sell_order_id=sell_id, ts=time.time())
        self.trades.append(t); self.last_traded_price = price; return t

class MCPLikePublisher:
    def __init__(self): self.events: List[str] = []
    def publish_snapshot(self, snap: Dict):
        import json; self.events.append(json.dumps({"type":"snapshot","data":snap}))
    def publish_trade(self, trade: Trade):
        import json; self.events.append(json.dumps({"type":"trade","data":{"price":trade.price,"qty":trade.qty,"ts":trade.ts,"buy_id":trade.buy_order_id,"sell_id":trade.sell_order_id}}))
"""
# code_path = "/Users/lucaslankry/Desktop/SKEMA/Semester 1 SKEMA/PYTHON/Projets/order_book/"
# with open(code_path, "w") as f:
#     f.write(module_src)

# Save a JSON log of events (snapshots + trades)
# log_path = "/mnt/data/order_book_demo_events.jsonl"
# with open(log_path, "w") as f:
#     for e in bus.events:
#         f.write(e + "\n")

# metrics_path = "/mnt/data/order_book_demo_metrics.json"
# with open(metrics_path, "w") as f:
#     json.dump(metrics, f, indent=2)

# (log_path, metrics_path, code_path, metrics)





