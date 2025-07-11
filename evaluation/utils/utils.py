import os
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pipeline.utils.general import load_json_data
from pipeline.utils.data_analysis_utils import get_pen_info

COLORS = {
    'upright': '#1B9E77',       # Dark Teal/Green
    'hanging': '#D95F02',       # Burnt Orange
    'difference': '#7570B3',    # Muted Purple
    'activity': '#E7298A',      
    'threshold': '#E6AB02',     # Dark Yellow/Mustard
    'warning': '#E6AB02',       # Alias for Threshold/Warning
    'critical': '#E41A1C',      # Red (Mean, Critical, TB Pens) - Standard Red
    'grid': '#D0D0D0',          # Lighter Gray for less intrusive grids
    'annotation': '#404040',    # Dark Gray for text/annotations (almost black)
    'background': 'white',      # Ensure white background
    'control': '#1B9E77',       # Explicit alias for Control pens
    'tail_biting': '#E41A1C',    # Explicit alias for TB pens
    'neutral': '#808080',       # Medium Gray
    'highlight': '#FEE08B',     # Light Yellow/Orange for highlighting bands
    'secondary_metric': '#377EB8', # Blue
}

PATTERN_COLORS = {
    #'Consistent-decline': COLORS['secondary_metric'],
    #'Inconsistent-decline': COLORS['neutral'],   
    #'Stable': COLORS['upright'], 
    #'Undefined': COLORS['critical']
    'Gleichmäßige Abnahme': COLORS['secondary_metric'],
    'Steile Abnahme': COLORS['neutral'],   
    'Stabil': COLORS['upright'], 
    'Undefiniert': COLORS['critical']
}


def set_plotting_style(config=None):
    """Set default matplotlib plotting style for dissertation quality."""
    if config is None: config = {}
    logger = logging.getLogger(__name__) # Get logger

    # --- Font Settings ---
    font_family = config.get('viz_font_family', 'serif')
    serif_fonts = config.get('viz_font_serif', [
        'Times New Roman', 'Times', 'DejaVu Serif',
        'Bitstream Vera Serif', 'Computer Modern Roman', 'serif'
    ])

    # Slightly larger base size for print readability
    font_base = config.get('viz_font_size_base', 11) # Default 11pt
    font_small = font_base - 1     # 10pt
    font_large = font_base + 1     # 12pt
    font_xlarge = font_base + 3    # 14pt

    # --- Resolution and Format ---
    # Higher DPI for print, consider PDF for vector graphics
    save_format = config.get('figure_format', 'png')
    save_dpi = config.get('figure_dpi', 600)
    logger.info(f"Setting plot style: Font={font_family}, BaseSize={font_base}pt, Format={save_format}, DPI={save_dpi}")
    if save_format.lower() in ['pdf', 'eps', 'svg']:
        logger.info("Using vector format for saving. Ensure fonts are embedded correctly if needed.")


    plt.rcParams.update({
        # --- Figure ---
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'figure.titlesize': font_xlarge,
        'figure.titleweight': 'bold', # Keep main title bold

        # --- Saving ---
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        'savefig.format': save_format,
        'savefig.dpi': save_dpi,
        'savefig.bbox': 'tight',      # Crucial for trimming whitespace
        'savefig.pad_inches': 0.05,   # Minimal padding

        # --- Font ---
        'font.family': font_family,
        'font.serif': serif_fonts,
        'font.size': font_base,
        'text.color': COLORS['annotation'], # Default text color
        # Ensure math text uses a compatible font if Times isn't fully supported
        'mathtext.fontset': 'stixsans', # 'stix' or 'cm' are other options

        # --- Axes ---
        'axes.facecolor': 'white',
        'axes.edgecolor': '#555555', # Slightly softer than black for spines
        'axes.linewidth': 0.8,       # Thinner axes lines
        'axes.titlesize': font_large,
        'axes.titleweight': 'bold', # Keep subplot titles bold
        'axes.labelsize': font_base,
        'axes.labelcolor': 'black', # Black labels for clarity
        'axes.labelweight': 'normal',
        'axes.titlepad': 8.0,       # Padding below subplot title
        'axes.labelpad': 5.0,       # Padding for axis labels

        # --- Spines ---
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,

        # --- Ticks ---
        'xtick.labelsize': font_small,
        'ytick.labelsize': font_small,
        'xtick.color': '#555555',    # Match axis edge color
        'ytick.color': '#555555',
        'xtick.major.size': 4,      # Slightly smaller ticks
        'ytick.major.size': 4,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,
        'xtick.direction': 'out',   # Ticks pointing out
        'ytick.direction': 'out',

        # --- Grid ---
        'axes.grid': True,          # Enable grid by default
        'grid.color': COLORS['grid'], # Lighter grid color
        'grid.linestyle': ':',      # Dotted grid lines
        'grid.linewidth': 0.6,      # Thinner grid lines
        'grid.alpha': 0.8,

        # --- Lines ---
        'lines.linewidth': 1.5,     # Slightly thinner lines overall
        'lines.markersize': 5,      # Default marker size
        'lines.markeredgewidth': 0.5,

        # --- Legend ---
        'legend.fontsize': font_small,
        'legend.frameon': True,     # Keep frame for clarity
        'legend.framealpha': 0.8,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#CCCCCC', # Light border for legend
        'legend.title_fontsize': font_base, # Legend title size

        # --- Boxplot / Violin ---
        'boxplot.showfliers': False, # Generally hide fliers for cleaner look in dissertations
        'boxplot.meanprops.marker': 'D', # Diamond for mean
        'boxplot.meanprops.markersize': 4,
        'boxplot.meanprops.markerfacecolor': COLORS['critical'], # Keep semantic color for mean
        'boxplot.meanprops.markeredgecolor': 'white', # White edge helps visibility
        'boxplot.boxprops.linewidth': 0.8,
        'boxplot.whiskerprops.linewidth': 0.8,
        'boxplot.whiskerprops.linestyle': '--',
        'boxplot.capprops.linewidth': 0.8,
        'boxplot.showmeans': True,
        'boxplot.patchartist': True, # Needed for coloring boxes
    })

# Helper function to lighten colors
def lighten_color(color, amount=0.4):
    """Lightens the given color by the given amount."""
    try:
        c = mcolors.to_rgb(color)
        import colorsys
        h, l, s = colorsys.rgb_to_hls(*c)
        new_l = l + (1 - l) * amount # Interpolate towards white
        return colorsys.hls_to_rgb(h, min(1, new_l), s=s)
    except ValueError:
        return color
    
# Helper function for saving processed data
def save_filtered_dataframes(filtered_results, config, logger, path_manager, pen_type_override=None):
    """
    Save filtered DataFrames to CSV files with organized directory structure.
    
    Args:
        filtered_results: List of filtered data dictionaries
        config: Configuration dictionary
        logger: Logger instance
        path_manager: Path manager instance
        pen_type_override: Optional string to force pen type classification
    """
    json_data = load_json_data(path_manager.path_to_piglet_rearing_info)
    
    # Get resample frequency from config for directory structure
    resample_freq = config.get('resample_freq', '1H')
    resample_freq_str = resample_freq.replace(':', '_').replace(' ', '_')  # Clean for filename
    
    # Create base directory structure
    base_output_dir = os.path.join(config.get('output_dir', '.'), 'filtered_data', resample_freq_str)
    
    # Define the DataFrame keys to save
    df_keys = ['raw_data', 'resampled_data', 'smoothed_data', 'interpolated_data']
    
    saved_count = 0
    skipped_count = 0
    
    for processed_data in filtered_results:
        camera = processed_data['camera']
        date_span = processed_data['date_span']
        
        # Get pen type (or use override)
        if pen_type_override:
            pen_type = pen_type_override
        else:
            pen_type, culprit_removal, datespan_gt = get_pen_info(camera, date_span, json_data)
        
        camera_label = camera.replace("Kamera", "Pen ").replace(" ", "_")
        
        # Map pen_type to directory name
        if pen_type == "tail biting":
            pen_type_dir = "tail_biting"
        elif pen_type == "control":
            pen_type_dir = "control"
        else:
            pen_type_dir = "other"
            
        # Create filename base (remove problematic characters)
        filename_base = f"{camera_label}_{date_span}".replace(" ", "_")
        
        # Save each DataFrame type
        for df_key in df_keys:
            df = processed_data.get(df_key)
            
            if df is not None and not df.empty:
                # Create directory structure
                output_dir = os.path.join(base_output_dir, pen_type_dir, df_key)
                os.makedirs(output_dir, exist_ok=True)
                
                # Create filename and path
                filename = f"{filename_base}_{df_key}.csv"
                filepath = os.path.join(output_dir, filename)
                
                try:
                    # Save DataFrame to CSV
                    df.to_csv(filepath, index=True)  # Save index (datetime) as well
                    saved_count += 1
                    logger.debug(f"Saved {df_key} for {camera}/{date_span} to {filepath}")
                except Exception as e:
                    logger.error(f"Failed to save {df_key} for {camera}/{date_span}: {e}")
                    skipped_count += 1
            else:
                logger.debug(f"Skipping empty/missing {df_key} for {camera}/{date_span}")
                skipped_count += 1
    
    logger.info(f"Saved {saved_count} DataFrames to {base_output_dir}")
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} empty/missing DataFrames")
    
    return base_output_dir