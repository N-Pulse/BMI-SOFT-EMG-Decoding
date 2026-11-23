import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_stacked_channels(df, time_col='Time', title='Channel Data'):
    """
    Plots each channel in its own subplot, sharing the X-axis.
    """
    # Identify data columns (exclude Time)
    data_cols = [c for c in df.columns if (c != time_col and "Ch" in c)]
    num_channels = len(data_cols)
    
    # Create subplots: rows=num_channels, cols=1, sharex=True locks the zoom
    fig, axes = plt.subplots(nrows=num_channels, ncols=1, figsize=(10, 2 * num_channels), sharex=True)
    
    # If there is only 1 channel, axes is not a list, so we wrap it
    if num_channels == 1:
        axes = [axes]

    for ax, col in zip(axes, data_cols):
        ax.plot(df[time_col], df[col], color='tab:blue', linewidth=0.8)
        ax.set_ylabel(col, rotation=0, labelpad=40, fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Remove x-ticks for all but the bottom plot to reduce clutter
        if ax != axes[-1]:
            ax.tick_params(labelbottom=False)

    # Set common X label on the bottom plot
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title, y=1.0) # Title at the top
    
    plt.tight_layout()
    plt.show()

def plot_true_vs_predicted(y_true, y_pred, title='True vs Predicted Labels'):
    """
    Plots true labels vs predicted labels as a function of sample index.
    
    Args:
        y_true: array of true labels
        y_pred: array of predicted labels
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.plot(y_true, label='True Labels', alpha=0.7, linewidth=1.5)
    ax.plot(y_pred, label='Predicted Labels', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Class Label')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

