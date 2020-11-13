import matplotlib.pyplot as plt

def plot_history(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [OP]')
  plt.legend()
  plt.grid(True)