import matplotlib.pyplot as plt
import numpy as np
import time

# Inizializza il grafico
plt.ion()  # Modalità interattiva
fig, ax = plt.subplots()
x_data, y_data = [], []
line, = ax.plot([], [], 'r-')  # Linea rossa

for i in range(100):
    x_data.append(i)
    y_data.append(np.sin(i * 0.1))  # Valore sinusoidale
    
    line.set_xdata(x_data)
    line.set_ydata(y_data)
    ax.relim()  # Aggiorna i limiti degli assi
    ax.autoscale_view()  # Ridimensiona gli assi
    plt.draw()  # Disegna il nuovo grafico
    plt.pause(0.05)  # Attendi un po' per simulare dati in tempo reale

plt.ioff()  # Disabilita la modalità interattiva
plt.show()