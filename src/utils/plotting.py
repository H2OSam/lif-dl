"""
Plotting utilities for LIF-DL visualization.

This module contains standardized plotting functions for creating figures
used in evaluation and analysis of lake ice forecasts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


def plot_variable_importance_grouped(vi_df, variable_order=None, variable_labels=None, save_path=None):
    """
    Create a grouped bar chart for variable importance with customizable ordering and labels.
    
    Parameters
    ----------
    vi_df : pd.DataFrame
        DataFrame containing variable importance data with columns 'Variable', 'Freezeup', 'Breakup'
    variable_order : list, optional
        Custom order for variables. If None, uses original order from DataFrame
    variable_labels : dict, optional
        Custom labels for variables. Keys should match variable names, values are display labels.
        If None, uses original variable names from DataFrame
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Set font to Times New Roman globally
    plt.rcParams['font.family'] = 'serif'
    
    # Apply custom ordering if provided
    if variable_order is not None:
        # Filter to only include variables that exist in the DataFrame
        available_vars = vi_df['Variable'].tolist()
        variable_order = [var for var in variable_order if var in available_vars]
        # Reorder the DataFrame
        vi_df = vi_df.set_index('Variable').reindex(variable_order).reset_index()
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Get data
    variables = vi_df['Variable']
    freezeup_vals = vi_df['Freezeup']
    breakup_vals = vi_df['Breakup']
    
    # Create display labels - use custom labels if provided, otherwise use original names
    if variable_labels is not None:
        display_labels = [variable_labels.get(var, var) for var in variables]
    else:
        display_labels = variables
    
    # Set up positions for grouped bars
    x = np.arange(len(variables))
    width = 0.35  # Width of bars
    
    # Create grouped bars
    bars1 = ax.bar(x - width/2, freezeup_vals, width, 
                   label='Freeze-up (Sep-Dec)', color='tab:blue', 
                   edgecolor='black', zorder=3)
    bars2 = ax.bar(x + width/2, breakup_vals, width, 
                   label='Break-up (Apr-Jul)', color='firebrick', 
                   edgecolor='black', zorder=3)
    
    # Customize the plot
    ax.set_ylabel('Relative Importance', fontsize=32, fontweight='bold')
    
    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=28)
    
    # Set y-axis
    ax.tick_params(axis='y', which='major', labelsize=28)
    ax.set_ylim(0, max(max(freezeup_vals), max(breakup_vals)) * 1.15)
    
    # Add gridlines
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(fontsize=28, loc='upper right', framealpha=0.95, edgecolor='black')
    
    # Add value labels on top of bars
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=16, rotation=0)
    
    add_value_labels(bars1, freezeup_vals)
    add_value_labels(bars2, breakup_vals)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_variable_importance_monthly_heatmap(vi_df, variable_order=None, variable_labels=None, save_path=None):
    """
    Create a monthly variable-importance heatmap (variables x months).

    Parameters
    ----------
    vi_df : pd.DataFrame
        DataFrame containing monthly variable importance in wide format with a
        'Variable' column and month columns.
    variable_order : list, optional
        Custom order for variables. If None, uses the order from DataFrame.
    variable_labels : dict, optional
        Custom display labels for variables. Keys should match variable names,
        values are display labels. If None, uses original variable names.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """
    plt.rcParams['font.family'] = 'serif'

    data = vi_df.copy()

    if 'Variable' in data.columns:
        data = data.set_index('Variable')
    else:
        data = data.set_index(data.columns[0])

    variables_to_exclude = ["lake_water", "lake_ice", "land"]
    data = data.drop(index=variables_to_exclude, errors='ignore')

    if variable_order is not None:
        variable_order = [var for var in variable_order if var in data.index]
        if variable_order:
            data = data.reindex(variable_order)

    month_order = ["Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]
    ordered_cols = [month for month in month_order if month in data.columns]
    data = data[ordered_cols]

    if variable_labels is not None:
        data = data.rename(index=variable_labels)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        data,
        cmap="cividis",
        vmin=0,
        vmax=0.50,
        linewidths=0.3,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
        cbar_kws={"label": "Relative Importance\n(normalized by month)", "pad": 0.02},
        ax=ax
    )

    if {'Sep', 'Dec', 'Apr', 'Jul'}.issubset(set(ordered_cols)):
        freeze_start = ordered_cols.index('Sep')
        freeze_end = ordered_cols.index('Dec')
        breakup_start = ordered_cols.index('Apr')
        breakup_end = ordered_cols.index('Jul')

        freeze_x0, freeze_x1 = freeze_start + 0.5, freeze_end + 0.5
        breakup_x0, breakup_x1 = breakup_start + 0.5, breakup_end + 0.5

        line_y = -0.08
        text_y = -0.10

        ax.plot([freeze_x0, freeze_x1], [line_y, line_y], transform=ax.get_xaxis_transform(), color='blue', lw=2, clip_on=False)
        ax.plot([breakup_x0, breakup_x1], [line_y, line_y], transform=ax.get_xaxis_transform(), color='red', lw=2, clip_on=False)

        ax.text(
            (freeze_start + freeze_end) / 2 + 0.5,
            text_y,
            'Freeze-up (Sep-Dec)',
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='top',
            fontsize=8,
            color='blue',
            clip_on=False
        )
        ax.text(
            (breakup_start + breakup_end) / 2 + 0.5,
            text_y,
            'Break-up (Apr-Jul)',
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='top',
            fontsize=8,
            color='red',
            clip_on=False
        )

    # ax.set_title("Monthly Relative Variable Importance", pad=12)
    ax.set_ylabel("")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout(rect=[0, 0.10, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_spatial_timing_maps(spatial_data, season='BUS', spatial_df=None, lake_order=None, save_path=None):
    """
    Create visualization of spatial timing anomaly maps.
    
    Parameters
    ----------
    spatial_data : dict
        Dictionary containing spatial data for each lake and source
    season : str
        'BUS' for Break-up Start or 'FUS' for Freeze-up Start
    spatial_df : pd.DataFrame, optional
        DataFrame containing SSIM and Kendall Tau-B metrics
    lake_order : list, optional
        Order of lakes for display. If None, uses keys from spatial_data.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Set font
    plt.rcParams['font.family'] = 'serif'
    
    # Zoom factors for each lake (to crop the edges)
    zooms = {
        "Great_Bear_Lake": 15,
        "Great_Slave_Lake": 25,
        "Lake_Athabasca": 18,
        "Reindeer_Lake": 25,
        "Lake_Winnipeg": 0
    }
    
    # Use provided lake order or default to spatial_data keys
    if lake_order is None:
        lake_order = list(spatial_data.keys())
    
    # Create figure: 3 rows (LIF-DL, IMS, FLake) × 5 columns (lakes)
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(18, 8),
                            constrained_layout=True, gridspec_kw={"wspace": 0.05})
    
    # Colormap for timing anomalies
    cmap = plt.get_cmap("RdBu_r")
    cmap.set_bad(color='gray', alpha=0.7)
    vmin, vmax = -14, 14
    
    # Source information
    sources = ["lif_dl", "ims", "flake"]
    source_labels = ["LIF-DL\n(proposed)", "IMS\n(observation)", "FLake\n(baseline)"]
    
    # Select the timing variable based on season
    timing_var = 'bus_timing' if season == 'BUS' else 'fus_timing'
    ssim_col = f'{season}_SSIM'
    ktb_col = f'{season}_Kendall_Tau_B'
    
    # Plot each source and lake
    for row, (source, label) in enumerate(zip(sources, source_labels)):
        for col, lake in enumerate(lake_order):
            ax = axes[row, col]
            z = zooms.get(lake, 0)
            
            # Get the data for this lake and source
            if lake not in spatial_data or source not in spatial_data[lake]:
                ax.axis("off")
                continue
            
            ds = spatial_data[lake][source]
            timing_map = ds[timing_var].values
            
            # Apply zoom (crop edges)
            if z > 0:
                timing_map = timing_map[z:-z, z:-z]
            
            # Plot
            im = ax.pcolormesh(timing_map, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Set border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.75)
            
            # Add titles and labels
            if row == 0:
                ax.set_title(lake.replace("_", " "), fontsize=22, weight="bold")
            if col == 0:
                ax.set_ylabel(label, fontsize=22, weight="bold")
            
            # Add metrics (only for LIF-DL and FLake rows)
            if spatial_df is not None and row != 1:  # Skip IMS row
                model_name = 'LIF-DL' if source == 'lif_dl' else 'FLAKE'
                
                metric_row = spatial_df[
                    (spatial_df['Lake'] == lake) &
                    (spatial_df['Model'] == model_name)
                ]
                
                if not metric_row.empty:
                    # Add SSIM on the left
                    if ssim_col in metric_row.columns:
                        ssim_val = metric_row[ssim_col].values[0]
                        if not np.isnan(ssim_val):
                            ax.text(0.04, 0.10, f"SSIM: {ssim_val:.2f}",
                                   transform=ax.transAxes,
                                   horizontalalignment='left', verticalalignment='top',
                                   fontsize=14, color='black')
                    
                    # Add KT-B on the right
                    if ktb_col in metric_row.columns:
                        ktb_val = metric_row[ktb_col].values[0]
                        if not np.isnan(ktb_val):
                            ax.text(0.64, 0.10, f"KT-B: {ktb_val:.2f}",
                                   transform=ax.transAxes,
                                   horizontalalignment='left', verticalalignment='top',
                                   fontsize=14, color='black')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.01)
    cbar.set_label("Timing Anomaly", fontsize=22)
    cbar.set_ticks(np.linspace(vmin, vmax, 5))
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.text(0.9, -0.05, "early", va='bottom', ha='center',
                fontsize=18, transform=cbar.ax.transAxes)
    cbar.ax.text(0.9, 1.05, "late", va='top', ha='center',
                fontsize=18, transform=cbar.ax.transAxes)
    
        
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_local_morans_i(spatial_data, season='BUS', spatial_df=None, lake_order=None, save_path=None):
    """
    Create visualization of Local Moran's I cluster maps.
    
    Parameters
    ----------
    spatial_data : dict
        Dictionary containing spatial data for each lake and source
    season : str
        'BUS' for Break-up Start or 'FUS' for Freeze-up Start
    spatial_df : pd.DataFrame, optional
        DataFrame containing LMI IoU metrics
    lake_order : list, optional
        Order of lakes for display. If None, uses keys from spatial_data.
    save_path : str, optional
        Save path for the figure. If None, figure is not saved.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Set font
    plt.rcParams['font.family'] = 'serif'
    
    # Zoom factors for each lake (to crop the edges)
    zooms = {
        "Great_Bear_Lake": 15,
        "Great_Slave_Lake": 25,
        "Lake_Athabasca": 18,
        "Reindeer_Lake": 25,
        "Lake_Winnipeg": 0
    }
    
    # Use provided lake order or default to spatial_data keys
    if lake_order is None:
        lake_order = list(spatial_data.keys())
    
    # Create figure: 3 rows (LIF-DL, IMS, FLake) × 5 columns (lakes)
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(18, 8), 
                             constrained_layout=True, gridspec_kw={"wspace": 0.05})
    
    # Create a custom discrete colormap for the quadrants
    cmap = ListedColormap(["#dddddd", "#d73027", "#fc8d59", "#4575b4", "#91bfdb"])
    cmap.set_bad(color='gray', alpha=0.7)  # Set color for NaNs
    vmin = 0
    vmax = 5
    
    # Source information
    sources = ["lif_dl", "ims", "flake"]
    source_labels = ["LIF-DL\n(proposed)", "IMS\n(observation)", "FLake\n(baseline)"]
    
    # Select the LMI cluster variable based on season
    lmi_var = 'bus_lmi_clusters' if season == 'BUS' else 'fus_lmi_clusters'
    iou_col = f'{season}_LMI_IoU'
    
    # Process each source and lake combination
    for row, (source, label) in enumerate(zip(sources, source_labels)):
        for col, lake in enumerate(lake_order):
            ax = axes[row, col]
            z = zooms.get(lake, 0)
            
            # Get the data for this lake and source
            if lake not in spatial_data or source not in spatial_data[lake]:
                ax.axis("off")
                continue
            
            ds = spatial_data[lake][source]
            cluster_map = ds[lmi_var].values
            
            # Apply zoom (crop edges)
            if z > 0:
                cluster_map = cluster_map[z:-z, z:-z]
            
            # Plot
            im = ax.pcolormesh(cluster_map, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Set the spines to be visible and thin
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.75)
            
            # Set titles and labels
            if row == 0:
                ax.set_title(lake.replace("_", " "), fontsize=22, weight="bold")
            if col == 0:
                ax.set_ylabel(label, fontsize=22, weight="bold")
                
            # Add IoU metric (only for LIF-DL and FLake rows)
            if spatial_df is not None and row != 1:  # Skip IMS row
                model_name = 'LIF-DL' if source == 'lif_dl' else 'FLAKE'
                
                metric_row = spatial_df[
                    (spatial_df['Lake'] == lake) &
                    (spatial_df['Model'] == model_name)
                ]
                
                if not metric_row.empty and iou_col in metric_row.columns:
                    iou_val = metric_row[iou_col].values[0]
                    if not np.isnan(iou_val):
                        ax.text(0.04, 0.10, f"IoU: {iou_val:.2f}",
                               transform=ax.transAxes,
                               horizontalalignment='left', verticalalignment='top',
                               fontsize=14, color='black')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.01)
    cbar.set_label("Local Moran's I Quadrant", fontsize=22, labelpad=10)
    ticks = [0.5, 1.5, 2.5, 3.5, 4.5]
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_ticklabels(['ns', 'HH', 'LH', 'LL', 'HL'])
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_spatial_model_error_maps(spatial_data, season='BUS', lake_order=None, save_path=None):
    """
    Create model-minus-IMS spatial timing error maps.

    Parameters
    ----------
    spatial_data : dict
        Dictionary containing spatial data for each lake and source.
    season : str
        'BUS' for Break-up Start or 'FUS' for Freeze-up Start.
    lake_order : list, optional
        Order of lakes for display. If None, uses keys from spatial_data.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """
    plt.rcParams['font.family'] = 'serif'

    zooms = {
        "Great_Bear_Lake": 15,
        "Great_Slave_Lake": 25,
        "Lake_Athabasca": 18,
        "Reindeer_Lake": 25,
        "Lake_Winnipeg": 0
    }

    if lake_order is None:
        lake_order = list(spatial_data.keys())

    fig, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(18, 5.5),
        constrained_layout=True,
        gridspec_kw={"wspace": 0.05}
    )

    cmap = plt.get_cmap("RdBu_r")
    cmap.set_bad(color='gray', alpha=0.7)
    vmin, vmax = -14, 14

    timing_var = 'bus_timing' if season == 'BUS' else 'fus_timing'
    sources = ["lif_dl", "flake"]
    source_labels = ["LIF-DL - IMS", "FLake - IMS"]

    im = None

    for row, (source, label) in enumerate(zip(sources, source_labels)):
        for col, lake in enumerate(lake_order):
            ax = axes[row, col]
            z = zooms.get(lake, 0)

            if (
                lake not in spatial_data or
                source not in spatial_data[lake] or
                "ims" not in spatial_data[lake]
            ):
                ax.axis("off")
                continue

            model_map = spatial_data[lake][source][timing_var].values
            ims_map = spatial_data[lake]["ims"][timing_var].values
            error_map = model_map - ims_map

            if z > 0:
                error_map = error_map[z:-z, z:-z]

            im = ax.pcolormesh(error_map, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])

            mae_days = np.nanmean(np.abs(error_map))
            if not np.isnan(mae_days):
                ax.text(
                    0.04,
                    0.10,
                    f"MAE: {mae_days:.2f} d",
                    transform=ax.transAxes,
                    horizontalalignment='left',
                    verticalalignment='top',
                    fontsize=14,
                    color='black'
                )

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.75)

            if row == 0:
                ax.set_title(lake.replace("_", " "), fontsize=20, weight="bold")
            if col == 0:
                ax.set_ylabel(label, fontsize=18, weight="bold")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.01)
        cbar.set_label("Model - IMS Timing Error", fontsize=18)
        cbar.set_ticks(np.linspace(vmin, vmax, 5))
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.text(
            0.9,
            -0.05,
            "earlier than IMS",
            va='bottom',
            ha='center',
            fontsize=12,
            transform=cbar.ax.transAxes
        )
        cbar.ax.text(
            0.9,
            1.05,
            "later than IMS",
            va='top',
            ha='center',
            fontsize=12,
            transform=cbar.ax.transAxes
        )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_fic_temporal_evolution(fic_data, lake_order, save_path=None):
    """
    Create visualization of Fraction Ice Cover temporal evolution.
    
    Parameters
    ----------
    fic_data : dict
        Dictionary containing FIC timeseries data for each lake and source.
        Keys are lake names, values are dicts with source names as keys.
    lake_order : list
        List of lakes in order to display
    save_path : str, optional
        Save path for the figure. If None, figure is not saved.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Set font
    plt.rcParams['font.family'] = 'serif'
    
    # Create figure: one row per lake
    n_lakes = len(lake_order)
    fig, axes = plt.subplots(nrows=n_lakes, ncols=1, figsize=(18, 12), 
                             layout='constrained', sharey=True, sharex=True)
    
    if n_lakes == 1:
        axes = [axes]
    
    # Per-source style dictionary (all params in one place)
    styles = {
        'ims': {
            'color': 'black',
            'label': 'IMS',
            'linestyle': '-',
            'linewidth': 2.0,
            'alpha': 0.7
        },
        'cis': {
            'color': "#2870DD",
            'label': 'CIS',
            'linestyle': '-',
            'linewidth': 2.0,
            'alpha': 0.7
        },
        'flake': {
            'color': "#6B13CF",
            'label': 'FLake',
            'linestyle': ':',
            'linewidth': 3.0,
            'alpha': 0.7
        },
        'lif_dl': {
            'color': '#d62728',
            'label': 'LIF-DL',
            'linestyle': '-.',
            'linewidth': 3.0,
            'alpha': 0.7
        }
    }
    # Map the labels
    labels = [v["label"] for k, v in styles.items()]

    # Plot each lake
    for i, lake in enumerate(lake_order):
        ax = axes[i]
        
        # Plot IMS first (baseline)
        if 'ims' in fic_data[lake]:
            df = fic_data[lake]['ims']
            ax.plot(df['Date'], df['FIC'], **styles['ims'])
        
        # Plot CIS if available
        if 'cis' in fic_data[lake]:
            df = fic_data[lake]['cis']
            ax.plot(df['Date'], df['FIC'], **styles['cis'])
        
        # Plot FLake
        if 'flake' in fic_data[lake]:
            df = fic_data[lake]['flake']
            ax.plot(df['Date'], df['FIC'], **styles['flake'])
        
        # Plot LIF-DL
        if 'lif_dl' in fic_data[lake]:
            df = fic_data[lake]['lif_dl']
            ax.plot(df['Date'], df['FIC'], **styles['lif_dl'])
        
        # Format axes
        ax.set_title(lake.replace("_", " "), loc="left", weight="bold", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(df['Date'].min(), df['Date'].max())
        ax.grid(True, alpha=0.3)
    
    # Set common labels
    fig.supxlabel("Date", weight="bold", fontsize=20, x=0.2)
    fig.supylabel("Ice Cover Fraction", weight="bold", fontsize=20, x=-0.03)
    
    # Add legend at the bottom
    fig.legend(labels, fontsize=18, loc='lower left', 
              bbox_to_anchor=(0.60, -0.02), ncol=4, 
              bbox_transform=fig.transFigure)
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
