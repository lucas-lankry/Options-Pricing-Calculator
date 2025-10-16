import pandas as pd 
import matplotlib.pyplot as plt 
import vectorbt as vbt 


platinium_data = vbt.YFData.download("PL=F", start="2022-01-01", end="2022-12-31", missing_index='drop').get('Close')
gold_data = vbt.YFData.download("GC=F", start="2022-01-01", end="2022-12-31", missing_index='drop').get('Close')
copper_data = vbt.YFData.download("HG=F", start="2022-01-01", end="2022-12-31", missing_index='drop').get('Close')


platinium_data.index = pd.to_datetime(platinium_data.index)
gold_data.index = pd.to_datetime(gold_data.index)
copper_data.index = pd.to_datetime(copper_data.index)

# fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(12,5))

# Plot each series on its own axis
# ax1.plot(platinium_data, label='Platinum Close', color='purple')
# ax1.set_ylabel('Platinum ($)')
# ax1.legend()
# ax1.grid(True)

fig, ax2= plt.subplots(figsize = (12,5))

ax2.plot(gold_data, label='Gold Close', color='gold')
ax2.set_ylabel('Gold ($)')
ax2.legend(loc = 'upper left')
ax2.grid(True)

ax3 = ax2.twinx()
copp = ax3.plot(copper_data.index, copper_data, label = 'Copper Close', color='orange' )
ax3.legend(loc = 'upper right')
# ax3.plot(copper_data, label='Copper Close', color='orange')
# ax3.set_ylabel('Copper ($)')
# ax3.legend()
# ax3.set_xlabel('Date')
# ax3.grid(True)

fig.suptitle('Gold and Copper Futures Close Prices (2022)', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # leaves room for suptitle
plt.show()



# plt.figure(figsize = (12,5))
# plt.plot(platinium_data)
# # plt.title('Platinum Futures Data', fontsize=16)
# plt.xlabel('Year', fontsize=15)
# plt.ylabel('Price ($)', fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.legend(['Close'], prop={'size': 15})

# ax = plt.figure(figsize=(12,5))


# Set the title and axis labels and sizes
# plt.title('Gold and Copper Futures Data', fontsize=16)
# ax.set_xlabel('Year-Month', fontsize=15)
# ax.set_ylabel('Gold Price ($)', fontsize=15)
# ax2.set_ylabel('Copper Price ($)', fontsize=15)
# ax.tick_params(axis='both', labelsize=15)
# ax2.tick_params(axis='y', labelsize=15)
# h1, l1 = ax.get_legend_handles_labels()
# h2, l2 = ax2.get_legend_handles_labels()
# ax.legend(h1+h2, l1+l2, loc=2, prop={'size': 15})


# plt.show()