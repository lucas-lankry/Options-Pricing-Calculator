import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

# Lire et préparer les données
df = pd.read_csv('spy_quotes.csv')
df.columns= ['date', 'open', 'high', 'low', 'close', 'volume']
df['date'] = pd.to_datetime(df['date'])

# # Calcul de la moyenne mobile 20 jours sur la colonne 'open'
# df['ma20'] = df['open'].rolling(window=20).mean()

# Affichage du début du DataFrame pour vérification
print(df.head(3))  

# Tracer les prix et la moyenne mobile
fig, ax = plt.subplots(figsize=(12, 6))


plt.plot(df.date, df.open, label='Prix d\'ouverture')
# plt.plot(df.date, df.volume, label='Volume', color='orange')
plt.xlabel('Year')

# Format de l'axe des dates
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

plt.xticks(rotation=45)
ax.set_xlabel('Mois')
plt.title('SP500 Prix et Volume')
plt.legend()
plt.tight_layout()
plt.show()