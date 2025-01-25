import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define configurable paths
BASE_DIR = '/Users/naveenmalla/Documents/Projects/Thesis/Images/Final_Results'
MODEL_DIR = 'vgae_best_model'

# Create full path and ensure directories exist
save_path = os.path.join(BASE_DIR, MODEL_DIR)
os.makedirs(save_path, exist_ok=True)

# Set style and theme
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_theme(style="darkgrid")
plt.rcParams['font.family'] = 'sans-serif'

# Data points
epochs = [ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
loss_values = [ 1.0367, 1.0329, 1.0305, 1.0294, 1.0280, 1.0278, 1.0272, 1.0270, 1.0268, 1.0268]
auc_values = [ 0.8162, 0.8169, 0.8172, 0.8173, 0.8170, 0.8179, 0.8174, 0.8178, 0.8179, 0.8179]
ap_values = [ 0.2233, 0.2239, 0.2240, 0.2238, 0.2236, 0.2246, 0.2238, 0.2246, 0.2244, 0.2245]

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
