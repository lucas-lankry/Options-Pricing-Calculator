# Projet : Simulateur de portefeuille multi-actifs avec optimisation dynamique et scénarios de stress


# Objectif

# Simuler un portefeuille avec plusieurs actifs financiers (actions, ETFs, crypto…).

# Optimiser dynamiquement la répartition tous les mois selon les rendements et volatilités historiques.

# Ajouter des scénarios de stress : chute de marché, volatilité extrême.

# Comparer la performance avec un portefeuille statique.



# import vectorbt as vbt 


# start_date = "2020-01-01"
# end_date = "2025-01-01"
# tickers = ['MSFT', 'AAPL']

# df_price = vbt.YFData.download(tickers, missing_index='drop', start = start_date ,end = end_date).get('Close')

# ma_30 = vbt.MA.run(df_price,window = 30)
# ma_50 = vbt.MA.run(df_price, window = 50)


# entries = ma_30.ma_crossed_above(ma_50)
# exits= ma_30.ma_crossed_below(ma_50)
# pf = vbt.Portfolio.from_signals(df_price,entries, exits)

# print(pf.stats())


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Draw market engine
market = mpatches.Rectangle((3.5, 4), 3, 1.5, edgecolor="black", facecolor="#f2f2f2", linewidth=2)
ax.add_patch(market)
ax.text(5, 4.75, "Moteur de marché\n(Order Book, Matching Engine)", 
        ha="center", va="center", fontsize=10, fontweight="bold")

# Draw MCP orchestrator
mcp = mpatches.Rectangle((3.5, 6), 3, 1, edgecolor="black", facecolor="#d0e1ff", linewidth=2)
ax.add_patch(mcp)
ax.text(5, 6.5, "Model Context Protocol (MCP)\nOrchestrateur & Interfaces", 
        ha="center", va="center", fontsize=10, fontweight="bold")

# Draw RL agents
rl1 = mpatches.Rectangle((0.5, 7.5), 2.5, 1, edgecolor="black", facecolor="#ffe6cc", linewidth=2)
ax.add_patch(rl1)
ax.text(1.75, 8, "Agent RL :\nMarket Maker", ha="center", va="center", fontsize=9)

rl2 = mpatches.Rectangle((0.5, 6), 2.5, 1, edgecolor="black", facecolor="#ffe6cc", linewidth=2)
ax.add_patch(rl2)
ax.text(1.75, 6.5, "Agent RL :\nMomentum Trader", ha="center", va="center", fontsize=9)

# Draw LLM agents
llm1 = mpatches.Rectangle((7.5, 7.5), 2.5, 1, edgecolor="black", facecolor="#e6ccff", linewidth=2)
ax.add_patch(llm1)
ax.text(8.75, 8, "Agent LLM :\nNews Trader", ha="center", va="center", fontsize=9)

llm2 = mpatches.Rectangle((7.5, 6), 2.5, 1, edgecolor="black", facecolor="#e6ccff", linewidth=2)
ax.add_patch(llm2)
ax.text(8.75, 6.5, "Agent LLM :\nMacro Trader", ha="center", va="center", fontsize=9)

# Draw classic agents
classic1 = mpatches.Rectangle((0.5, 4.5), 2.5, 1, edgecolor="black", facecolor="#ccffcc", linewidth=2)
ax.add_patch(classic1)
ax.text(1.75, 5, "Agent Classique :\nArbitrage", ha="center", va="center", fontsize=9)

classic2 = mpatches.Rectangle((7.5, 4.5), 2.5, 1, edgecolor="black", facecolor="#ccffcc", linewidth=2)
ax.add_patch(classic2)
ax.text(8.75, 5, "Agent Classique :\nNoise Trader", ha="center", va="center", fontsize=9)

# Draw arrows
arrowprops = dict(arrowstyle="->", color="black", linewidth=1.5)

# RL to MCP
ax.annotate("", xy=(3.5, 8), xytext=(2.9, 8), arrowprops=arrowprops)
ax.annotate("", xy=(3.5, 6.5), xytext=(2.9, 6.5), arrowprops=arrowprops)

# LLM to MCP
ax.annotate("", xy=(7.5, 8), xytext=(7.1, 8), arrowprops=arrowprops)
ax.annotate("", xy=(7.5, 6.5), xytext=(7.1, 6.5), arrowprops=arrowprops)

# Classic to MCP
ax.annotate("", xy=(3.5, 5), xytext=(2.9, 5), arrowprops=arrowprops)
ax.annotate("", xy=(7.5, 5), xytext=(7.1, 5), arrowprops=arrowprops)

# MCP to Market
ax.annotate("", xy=(5, 6), xytext=(5, 5.5), arrowprops=arrowprops)

# Market output
ax.annotate("Prix, Volatilité,\nSpreads, Liquidité", 
            xy=(5, 4), xytext=(5, 2.5),
            ha="center", va="center",
            arrowprops=arrowprops, fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="black"))

# Final touches
ax.set_xlim(0, 11)
ax.set_ylim(2, 9)
ax.axis("off")
ax.set_title("Schéma du projet : Marché multi-agents (RL + LLM + Classiques) orchestré par MCP", fontsize=12, fontweight="bold")

plt.show()
