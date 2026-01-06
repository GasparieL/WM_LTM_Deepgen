import pandas as pd
import seaborn as sns
import numpy as np
import os.path as op
import glob
import matplotlib.pyplot as plt
import ast
# import moss
import csv
import random
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import os
from pandas import Timestamp
from datetime import timedelta
from sklearn.preprocessing import StandardScaler

def combined_rolling_average_and_model_fit(df, x_col="Tested - Untested IT Distractor Similarity", 
                                           y_col='Accuracy', item_col=None,
                                           window_percent=0.2, step_percent=0.01,
                                           palette=None):
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Remove grid
    ax.grid(False)
    
    # Use provided palette or default to a predefined one
    # if palette is None:
        # Default seaborn palette if none is provided
        # palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Color map for specific item types (using palette colors)
    color_map = {
        'prioritized': palette[0],
        'deprioritized': palette[1]
    }
    
    # If splitting by item_col
    if item_col is not None and item_col in df.columns:
        # Get unique tested items
        tested_items = df[item_col].unique()
        
        # First, plot rolling averages (in background)
        for i, item in enumerate(tested_items):
            df_item = df[df[item_col] == item].sort_values(by=x_col).reset_index(drop=True)
            
            total_points = len(df_item)
            
            window_points = max(int(total_points * window_percent), 3)  # Ensure at least 3 points
            step_points = max(int(total_points * step_percent), 1)      # Ensure at least 1 point
            
            if total_points < window_points:
                print(f"Skipping {item} - insufficient data points ({total_points} < {window_points})")
                continue
                
            print(f"Item: {item}, Total points: {total_points}, Window size: {window_points}, Step size: {step_points}")
                
            x_values = []
            y_values = []
            y_errors = []
            
            for j in range(0, total_points - window_points + 1, step_points):
                window = df_item.iloc[j:j+window_points]
                
                mean_x = window[x_col].mean()
                mean_y = window[y_col].mean()
                
                std_y = window[y_col].std()
                sem_y = std_y / np.sqrt(window_points)
                
                x_values.append(mean_x)
                y_values.append(mean_y)
                y_errors.append(sem_y)
            
            x_values = np.array(x_values)
            y_values = np.array(y_values)
            y_errors = np.array(y_errors)
            
            if len(x_values) == 0:
                continue
            
            # Select color based on item name or index in palette
            item_lower = str(item).lower()
            if item_lower in color_map:
                color = color_map[item_lower]
            elif 'prioritized' in item_lower:
                color = color_map['prioritized']
            elif 'deprioritized' in item_lower:
                color = color_map['deprioritized']
            else:
                # Use palette color with modulo to ensure we don't go out of bounds
                color = palette[i % len(palette)]
            
            # Very transparent rolling average lines
            ax.plot(x_values, y_values, color=color, linewidth=1, alpha=0.1, label=f'{item} (rolling average)')
            
            # Very transparent confidence bands
            ax.fill_between(x_values, 
                           y_values - 1.96 * y_errors,
                           y_values + 1.96 * y_errors,
                           color=color, alpha=0.1)
        
        # Then plot model fits (bold lines on top)
        for i, item in enumerate(tested_items):
            df_item = df[df[item_col] == item]
            
            # Select color based on item name or index in palette (same logic as above)
            item_lower = str(item).lower()
            if item_lower in color_map:
                color = color_map[item_lower]
            elif 'prioritized' in item_lower:
                color = color_map['prioritized']
            elif 'deprioritized' in item_lower:
                color = color_map['deprioritized']
            else:
                color = palette[i % len(palette)]
                
            # Fit a 2nd order polynomial
            x_range = np.linspace(df_item[x_col].min(), df_item[x_col].max(), 100)
            model = np.poly1d(np.polyfit(df_item[x_col], df_item[y_col], 2))
            
            # Very bold model fit lines
            ax.plot(x_range, model(x_range), color=color, linestyle='-', 
                    linewidth=5, label=f'{item} (model fit)')
        
        title_suffix = f" by {item_col}"
    
    # If NOT splitting by item_col
    else:
        # Sort by x value
        df_sorted = df.sort_values(by=x_col).reset_index(drop=True)
        
        total_points = len(df_sorted)
        
        window_points = max(int(total_points * window_percent), 3)  # Ensure at least 3 points
        step_points = max(int(total_points * step_percent), 1)      # Ensure at least 1 point
        
        print(f"Total points: {total_points}, Window size: {window_points}, Step size: {step_points}")
            
        x_values = []
        y_values = []
        y_errors = []
        
        for j in range(0, total_points - window_points + 1, step_points):
            window = df_sorted.iloc[j:j+window_points]
            
            mean_x = window[x_col].mean()
            mean_y = window[y_col].mean()
            
            std_y = window[y_col].std()
            sem_y = std_y / np.sqrt(window_points)
            
            x_values.append(mean_x)
            y_values.append(mean_y)
            y_errors.append(sem_y)
        
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        y_errors = np.array(y_errors)
        
        # Use first color from palette
        color = palette[0]
        
        # Very transparent rolling average line
        ax.plot(x_values, y_values, color=color, linewidth=1, alpha=0.9, label='Rolling average')
        
        # Very transparent confidence bands
        ax.fill_between(x_values, 
                       y_values - 1.96 * y_errors,
                       y_values + 1.96 * y_errors,
                       color=color, alpha=0.1)
        
        # Fit a 2nd order polynomial to all data
        x_range = np.linspace(df[x_col].min(), df[x_col].max(), 100)
        model = np.poly1d(np.polyfit(df[x_col], df[y_col], 2))
        
        # Bold model fit line
        ax.plot(x_range, model(x_range), color=color, linestyle='-', 
                linewidth=5, label='Model fit')
                
        title_suffix = ""
    
    # Styling for axes and labels
    ax.set_xlabel(x_col, fontweight='bold')
    ax.set_ylabel(y_col, fontweight='bold')
    ax.set_title(f'Rolling Average and Model Fit{title_suffix}', fontweight='bold')
    
    # Create legend and position it OUTSIDE the plot
    legend = ax.legend(frameon=True, fontsize=10, loc='center left', bbox_to_anchor=(1.0, 0.5))
    
    # Set y-axis limits if applicable
    if y_col.lower() in ['accuracy', 'acc', 'correct', 'resp_correct']:
        ax.set_ylim(0.65, 0.8)
    
    # Add spines with minimal styling
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)  # Make room for the legend on the right
    
    return plt
# def combined_sliding_window_and_model_fit_split(df, x_col="Tested - Untested V2 Distractor Similarity", 
#                                                y_col='Accuracy', item_col=None,
#                                                split_col='v2_pos_neg', split_value=0,
#                                                window_percent=0.2, step_percent=0.02,
#                                                palette=None, aspect=0.7, figsize_height=6):
#     """
#     Creates a two-panel plot with data split at split_value of x_col.
#     Shows rolling averages and polynomial fits, with option to group by item_col.
#     """
#     # Set up default palette if none provided
#     if palette is None:
#         default_colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
#     else:
#         default_colors = palette
    
#     # Color map for specific items
#     color_map = {
#         'prioritized': default_colors[0],    
#         'deprioritized': default_colors[1],  
#     }
    
#     # Split the data
#     df_left = df[df[x_col] <= split_value].copy()
#     df_right = df[df[x_col] > split_value].copy()
    
#     # Calculate figure width based on height and aspect ratio
#     figsize_width = 2 * figsize_height * aspect  # 2 panels * height * aspect
    
#     # Create figure with specified dimensions
#     fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(figsize_width, figsize_height), sharey=True)
#     fig.patch.set_facecolor('white')
#     ax_left.set_facecolor('white')
#     ax_right.set_facecolor('white')
    
#     # Function to process data for each side
#     def process_side(df_side, ax, side_name, group_by_item=True):
#         if group_by_item and item_col is not None and item_col in df.columns:
#             # Process by item groups
#             items = df_side[item_col].unique()
            
#             # First, plot rolling averages (in background)
#             for i, item in enumerate(items):
#                 df_item = df_side[df_side[item_col] == item].sort_values(by=x_col).reset_index(drop=True)
                
#                 total_points = len(df_item)
                
#                 # Calculate window and step sizes for THIS SPECIFIC ITEM
#                 window_points = max(int(total_points * window_percent), 3)  # Ensure at least 3 points
#                 step_points = max(int(total_points * step_percent), 1)      # Ensure at least 1 point
                
#                 # Set color based on item
#                 item_lower = str(item).lower()
#                 if item_lower in color_map:
#                     color = color_map[item_lower]
#                 elif 'prioritized' in item_lower:
#                     color = color_map['prioritized']
#                 elif 'deprioritized' in item_lower:
#                     color = color_map['deprioritized']
#                 else:
#                     color = default_colors[i % len(default_colors)]
                
#                 # Add rolling average if enough points
#                 if total_points >= window_points:
#                     print(f"{side_name} side - {item}, Total points: {total_points}, "
#                           f"Window size: {window_points}, Step size: {step_points}")
                        
#                     x_vals, y_vals, y_errs = [], [], []
                    
#                     for j in range(0, total_points - window_points + 1, step_points):
#                         window = df_item.iloc[j:j+window_points]
                        
#                         mean_x = window[x_col].mean()
#                         mean_y = window[y_col].mean()
#                         std_y = window[y_col].std()
#                         sem_y = std_y / np.sqrt(window_points)
                        
#                         x_vals.append(mean_x)
#                         y_vals.append(mean_y)
#                         y_errs.append(sem_y)
                    
#                     x_vals = np.array(x_vals)
#                     y_vals = np.array(y_vals)
#                     y_errs = np.array(y_errs)
                    
#                     if len(x_vals) > 0:
#                         # Very transparent rolling average
#                         ax.plot(x_vals, y_vals, color=color, linewidth=1, alpha=0.3, 
#                                label=f'{item} (rolling)')
                        
#                         # Very transparent confidence bands
#                         ax.fill_between(x_vals, 
#                                        y_vals - 1.96 * y_errs,
#                                        y_vals + 1.96 * y_errs,
#                                        color=color, alpha=0.1)
            
#             # Then plot model fits (bold lines on top)
#             for i, item in enumerate(items):
#                 df_item = df_side[df_side[item_col] == item]
#                 total_points = len(df_item)
                
#                 # Set color based on item
#                 item_lower = str(item).lower()
#                 if item_lower in color_map:
#                     color = color_map[item_lower]
#                 elif 'prioritized' in item_lower:
#                     color = color_map['prioritized']
#                 elif 'deprioritized' in item_lower:
#                     color = color_map['deprioritized']
#                 else:
#                     color = default_colors[i % len(default_colors)]
                
#                 # Add model fit if enough points
#                 if total_points >= 5:  # Need at least 5 points for a reasonable quadratic fit
#                     # Fit a 2nd order polynomial
#                     x_range = np.linspace(df_item[x_col].min(), df_item[x_col].max(), 100)
#                     model = np.poly1d(np.polyfit(df_item[x_col], df_item[y_col], 2))
                    
#                     # Bold model fit lines
#                     ax.plot(x_range, model(x_range), color=color, linestyle='-', 
#                             linewidth=5, label=f'{item} (model)')
#         else:
#             # Process all data as one group
#             df_side = df_side.sort_values(by=x_col).reset_index(drop=True)
#             total_points = len(df_side)
            
#             # Calculate window and step sizes for ALL DATA
#             window_points = max(int(total_points * window_percent), 3)  # Ensure at least 3 points
#             step_points = max(int(total_points * step_percent), 1)      # Ensure at least 1 point
            
#             color = default_colors[0]
            
#             # First plot rolling average (background)
#             if total_points >= window_points:
#                 print(f"{side_name} side - All Data, Total points: {total_points}, "
#                       f"Window size: {window_points}, Step size: {step_points}")
                    
#                 x_vals, y_vals, y_errs = [], [], []
                
#                 for j in range(0, total_points - window_points + 1, step_points):
#                     window = df_side.iloc[j:j+window_points]
                    
#                     mean_x = window[x_col].mean()
#                     mean_y = window[y_col].mean()
#                     std_y = window[y_col].std()
#                     sem_y = std_y / np.sqrt(window_points)
                    
#                     x_vals.append(mean_x)
#                     y_vals.append(mean_y)
#                     y_errs.append(sem_y)
                
#                 x_vals = np.array(x_vals)
#                 y_vals = np.array(y_vals)
#                 y_errs = np.array(y_errs)
                
#                 if len(x_vals) > 0:
#                     # Very transparent rolling average
#                     ax.plot(x_vals, y_vals, color=color, linewidth=1, alpha=0.3, 
#                            label='Rolling average')
                    
#                     # Very transparent confidence bands
#                     ax.fill_between(x_vals, 
#                                    y_vals - 1.96 * y_errs,
#                                    y_vals + 1.96 * y_errs,
#                                    color=color, alpha=0.1)
            
#             # Then plot model fit (bold, on top)
#             if total_points >= 5:  # Need at least 5 points for a reasonable quadratic fit
#                 # Fit a 2nd order polynomial
#                 x_range = np.linspace(df_side[x_col].min(), df_side[x_col].max(), 100)
#                 model = np.poly1d(np.polyfit(df_side[x_col], df_side[y_col], 2))
                
#                 # Bold model fit line
#                 ax.plot(x_range, model(x_range), color=color, linestyle='-', 
#                         linewidth=5, label='Model fit')
    
#     # Process both sides
#     use_item_col = (item_col is not None and item_col in df.columns)
#     process_side(df_left, ax_left, "Negative", use_item_col)
#     process_side(df_right, ax_right, "Positive", use_item_col)
    
#     # Set titles and labels
#     ax_left.set_title(f"{x_col} ≤ {split_value}")
#     ax_right.set_title(f"{x_col} > {split_value}")
    
#     ax_left.set_xlabel(x_col)
#     ax_right.set_xlabel('')  # Remove x-axis label from second subplot
    
#     ax_left.set_ylabel(y_col)
    
#     # Style the axes
#     ax_right.yaxis.set_visible(False)
#     ax_right.spines['left'].set_visible(False)
#     ax_left.spines['right'].set_visible(False)
#     ax_right.spines['right'].set_visible(False)
#     ax_right.spines['top'].set_visible(False)
#     ax_left.spines['top'].set_visible(False)
    
#     # Remove grid
#     ax_left.grid(False)
#     ax_right.grid(False)
    
#     # Add legend to right plot - only keep unique items by using set()
#     by_label = {}
#     for ax in [ax_left, ax_right]:
#         handles, labels = ax.get_legend_handles_labels()
#         # Group handles by labels and take the first handle for each label
#         for h, lbl in zip(handles, labels):
#             # Just use the item name without the "(rolling)" or "(model)" suffix
#             if "(" in lbl:
#                 base_label = lbl.split("(")[0].strip()
#             else:
#                 base_label = lbl
#             if base_label not in by_label:
#                 by_label[base_label] = h
    
#     # Place legend outside and below the plot
#     if by_label:
#         fig.legend(by_label.values(), by_label.keys(), 
#                   loc='lower center', ncol=len(by_label), bbox_to_anchor=(0.5, -0.05))
    
#     # Set y-limits for accuracy plots
#     if y_col.lower() in ['accuracy', 'acc', 'correct', 'resp_correct']:
#         y_min = 0.65
#         y_max = 0.8
#         ax_left.set_ylim(y_min, y_max)
    
#     # Add main title
#     window_percent_text = f"{window_percent*100:.0f}%"
#     title_suffix = f" by {item_col}" if use_item_col else ""
#     fig.suptitle(f'Rolling Average and Model Fit{title_suffix}', 
#                  fontweight='bold', fontsize=14, y=0.98)
    
#     plt.tight_layout()
#     plt.subplots_adjust(bottom=0.15)  # Make room for the legend
#     return fig
def combined_sliding_window_and_model_fit_split_col(df, x_col="Tested - Untested V2 Distractor Similarity", 
                                               y_col='Accuracy', item_col=None,
                                               split_col='v2_pos_neg', split_value=0,
                                               window_percent=0.2, step_percent=0.02,
                                               palette=None, aspect=0.7, figsize_height=6):
    """
    Creates a two-panel plot with data split at split_value of x_col.
    Shows rolling averages and polynomial fits, with option to group by item_col.
    """
    # Set up default palette if none provided
    if palette is None:
        default_colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
    else:
        default_colors = palette
    
    # Color map for specific items
    color_map = {
        'prioritized': default_colors[0],    
        'deprioritized': default_colors[1],  
    }
    
    # Split the data
    df_left = df[df[x_col] <= split_value].copy()
    df_right = df[df[x_col] > split_value].copy()
    
    # Calculate figure width based on height and aspect ratio
    figsize_width = 2 * figsize_height * aspect  # 2 panels * height * aspect
    
    # Create figure with specified dimensions
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(figsize_width, figsize_height), sharey=True)
    fig.patch.set_facecolor('white')
    ax_left.set_facecolor('white')
    ax_right.set_facecolor('white')
    
    # Create a consistent item to color mapping using ALL data (not just one side)
    item_color_dict = {}
    if item_col is not None and item_col in df.columns:
        # Get all unique items from the entire dataset
        all_items = df[item_col].unique()
        
        # Create color mapping for all items
        for i, item in enumerate(all_items):
            item_lower = str(item).lower()
            if item_lower in color_map:
                item_color_dict[item] = color_map[item_lower]
            elif 'prioritized' in item_lower:
                item_color_dict[item] = color_map['prioritized']
            elif 'deprioritized' in item_lower:
                item_color_dict[item] = color_map['deprioritized']
            else:
                item_color_dict[item] = default_colors[i % len(default_colors)]
    
    # Function to process data for each side
    def process_side(df_side, ax, side_name, group_by_item=True):
        if group_by_item and item_col is not None and item_col in df.columns:
            # Process by item groups
            items = df_side[item_col].unique()
            
            # First, plot rolling averages (in background)
            for i, item in enumerate(items):
                df_item = df_side[df_side[item_col] == item].sort_values(by=x_col).reset_index(drop=True)
                
                total_points = len(df_item)
                
                # Calculate window and step sizes for THIS SPECIFIC ITEM
                window_points = max(int(total_points * window_percent), 3)  # Ensure at least 3 points
                step_points = max(int(total_points * step_percent), 1)      # Ensure at least 1 point
                
                # Get color from the consistent global mapping
                color = item_color_dict.get(item, default_colors[i % len(default_colors)])
                
                # Add rolling average if enough points
                if total_points >= window_points:
                    print(f"{side_name} side - {item}, Total points: {total_points}, "
                          f"Window size: {window_points}, Step size: {step_points}")
                        
                    x_vals, y_vals, y_errs = [], [], []
                    
                    for j in range(0, total_points - window_points + 1, step_points):
                        window = df_item.iloc[j:j+window_points]
                        
                        mean_x = window[x_col].mean()
                        mean_y = window[y_col].mean()
                        std_y = window[y_col].std()
                        sem_y = std_y / np.sqrt(window_points)
                        
                        x_vals.append(mean_x)
                        y_vals.append(mean_y)
                        y_errs.append(sem_y)
                    
                    x_vals = np.array(x_vals)
                    y_vals = np.array(y_vals)
                    y_errs = np.array(y_errs)
                    
                    if len(x_vals) > 0:
                        # Very transparent rolling average
                        ax.plot(x_vals, y_vals, color=color, linewidth=1, alpha=0.5, 
                               label=f'{item} (rolling)')
                        
                        # Very transparent confidence bands
                        ax.fill_between(x_vals, 
                                       y_vals - 1.96 * y_errs,
                                       y_vals + 1.96 * y_errs,
                                       color=color, alpha=0.3)
            
            # Then plot model fits (bold lines on top)
            for i, item in enumerate(items):
                df_item = df_side[df_side[item_col] == item]
                total_points = len(df_item)
                
                # Get color from the consistent global mapping
                color = item_color_dict.get(item, default_colors[i % len(default_colors)])
                
                # Add model fit if enough points
                if total_points >= 5:  # Need at least 5 points for a reasonable quadratic fit
                    # Fit a 2nd order polynomial
                    x_range = np.linspace(df_item[x_col].min(), df_item[x_col].max(), 100)
                    model = np.poly1d(np.polyfit(df_item[x_col], df_item[y_col], 2))
                    
                    # Bold model fit lines
                    ax.plot(x_range, model(x_range), color=color, linestyle='-', 
                            linewidth=5, label=f'{item} (model)')
        else:
            # Process all data as one group
            df_side = df_side.sort_values(by=x_col).reset_index(drop=True)
            total_points = len(df_side)
            
            # Calculate window and step sizes for ALL DATA
            window_points = max(int(total_points * window_percent), 3)  # Ensure at least 3 points
            step_points = max(int(total_points * step_percent), 1)      # Ensure at least 1 point
            
            color = default_colors[0]
            
            # First plot rolling average (background)
            if total_points >= window_points:
                print(f"{side_name} side - All Data, Total points: {total_points}, "
                      f"Window size: {window_points}, Step size: {step_points}")
                    
                x_vals, y_vals, y_errs = [], [], []
                
                for j in range(0, total_points - window_points + 1, step_points):
                    window = df_side.iloc[j:j+window_points]
                    
                    mean_x = window[x_col].mean()
                    mean_y = window[y_col].mean()
                    std_y = window[y_col].std()
                    sem_y = std_y / np.sqrt(window_points)
                    
                    x_vals.append(mean_x)
                    y_vals.append(mean_y)
                    y_errs.append(sem_y)
                
                x_vals = np.array(x_vals)
                y_vals = np.array(y_vals)
                y_errs = np.array(y_errs)
                
                if len(x_vals) > 0:
                    # Very transparent rolling average
                    ax.plot(x_vals, y_vals, color=color, linewidth=1, alpha=0.5, 
                           label='Rolling average')
                    
                    # Very transparent confidence bands
                    ax.fill_between(x_vals, 
                                   y_vals - 1.96 * y_errs,
                                   y_vals + 1.96 * y_errs,
                                   color=color, alpha=0.3)
            
            # Then plot model fit (bold, on top)
            if total_points >= 5:  # Need at least 5 points for a reasonable quadratic fit
                # Fit a 2nd order polynomial
                x_range = np.linspace(df_side[x_col].min(), df_side[x_col].max(), 100)
                model = np.poly1d(np.polyfit(df_side[x_col], df_side[y_col], 2))
                
                # Bold model fit line
                ax.plot(x_range, model(x_range), color=color, linestyle='-', 
                        linewidth=5, label='Model fit')
    
    # Process both sides
    use_item_col = (item_col is not None and item_col in df.columns)
    process_side(df_left, ax_left, "Negative", use_item_col)
    process_side(df_right, ax_right, "Positive", use_item_col)
    
    # Set titles and labels
    ax_left.set_title(f"{x_col} ≤ {split_value}")
    ax_right.set_title(f"{x_col} > {split_value}")
    
    ax_left.set_xlabel(x_col)
    ax_right.set_xlabel('')  # Remove x-axis label from second subplot
    
    ax_left.set_ylabel(y_col)
    
    # Style the axes
    ax_right.yaxis.set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['top'].set_visible(False)
    ax_left.spines['top'].set_visible(False)
    
    # Remove grid
    ax_left.grid(False)
    ax_right.grid(False)
    
    # Add legend to right plot - only keep unique items by using set()
    by_label = {}
    for ax in [ax_left, ax_right]:
        handles, labels = ax.get_legend_handles_labels()
        # Group handles by labels and take the first handle for each label
        for h, lbl in zip(handles, labels):
            # Just use the item name without the "(rolling)" or "(model)" suffix
            if "(" in lbl:
                base_label = lbl.split("(")[0].strip()
            else:
                base_label = lbl
            if base_label not in by_label:
                by_label[base_label] = h
    
    # Place legend outside and below the plot
    if by_label:
        fig.legend(by_label.values(), by_label.keys(), 
                  loc='lower center', ncol=len(by_label), bbox_to_anchor=(0.5, -0.05))
    
    # Set y-limits for accuracy plots
    if y_col.lower() in ['accuracy', 'acc', 'correct', 'resp_correct']:
        y_min = 0.65
        y_max = 0.8
        ax_left.set_ylim(y_min, y_max)
    
    # Add main title
    window_percent_text = f"{window_percent*100:.0f}%"
    title_suffix = f" by {item_col}" if use_item_col else ""
    fig.suptitle(f'Rolling Average and Model Fit{title_suffix}', 
                 fontweight='bold', fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend
    return fig