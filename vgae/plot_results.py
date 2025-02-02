import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define configurable paths
BASE_DIR = '/Users/naveenmalla/Documents/Projects/Thesis/Images/Final_Results'
MODEL_DIR = 'vgae_best_model_100_500'

# Create full path and ensure directories exist
save_path = os.path.join(BASE_DIR, MODEL_DIR)
os.makedirs(save_path, exist_ok=True)

# Set style and theme
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_theme(style="darkgrid")
plt.rcParams['font.family'] = 'sans-serif'

# Data points. The values here are hardcoded because the training code did not have the capability to plot the metrics. it printed the values to terminal and the values were saved after training each of these models.
# Since retraining the model will take a lot of time, I have hardcoded the values here.
epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
loss_values = [0.9916, 0.9846, 0.9797, 0.9780, 0.9776, 0.9768, 0.9765, 0.9760, 0.9758, 0.9755]
auc_values = [0.9079, 0.9155, 0.9210, 0.9209, 0.9209, 0.9217, 0.9220, 0.9218, 0.9221, 0.9222]
ap_values = [0.0969, 0.1047, 0.1111, 0.1109, 0.1109, 0.1120, 0.1122, 0.1120, 0.1125, 0.1125]

# Plot 1: Loss Progression (Horizontal)
plt.figure(figsize=(15, 5))
plt.plot(epochs, loss_values, color='#E74C3C', linewidth=2.5, marker='o', 
         markersize=6, markeredgecolor='white', markeredgewidth=1.5,
         label='Training Loss')
plt.title('Loss Progression', fontsize=14, pad=15)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(epochs) 
plt.legend(loc='upper right')

plt.tight_layout()
# Save loss plot
plt.savefig(os.path.join(save_path, MODEL_DIR +'_loss.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: AUC and AP side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.set_xticks(epochs)  
ax2.set_xticks(epochs)

# AUC plot
ax1.plot(epochs, auc_values, color='#2ECC71', linewidth=2.5, marker='o',
         markersize=6, markeredgecolor='white', markeredgewidth=1.5,
         label='Training AUC')
ax1.set_title('AUC Progression', fontsize=14, pad=15)
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('AUC Score', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower right')

# AP plot
ax2.plot(epochs, ap_values, color='#3498DB', linewidth=2.5, marker='o',
         markersize=6, markeredgecolor='white', markeredgewidth=1.5,
         label='Training AP')
ax2.set_title('AP Progression', fontsize=14, pad=15)
ax2.set_xlabel('Epochs', fontsize=12)
ax2.set_ylabel('AP Score', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right')

plt.tight_layout()
# Save metrics plot
plt.savefig(os.path.join(save_path, MODEL_DIR + '_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()
