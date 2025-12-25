"""
UI Constants for Publication-Ready Molecular Dynamics Simulator
Defines consistent styling, fonts, colors, and spacing for clean, professional appearance.
"""

# Font Configuration
FONT_FAMILY = "sans-serif"  # Cross-platform generic font family
FONT_SIZES = {
    'title': 14,
    'subtitle': 11,
    'label': 10,
    'caption': 9,
    'small': 8,
}

# Color Scheme (Publication-friendly)
COLORS = {
    'background': '#ffffff',  # Clean white background
    'panel_bg': '#f8f8f8',    # Light gray panels
    'text_primary': '#2c2c2c',  # Dark gray text
    'text_secondary': '#666666', # Medium gray text
    'accent': '#2563eb',      # Professional blue
    'accent_light': '#dbeafe', # Light blue for highlights
    'border': '#e5e5e5',      # Light border
    'grid': '#f0f0f0',        # Very light grid lines
    'success': '#16a34a',     # Green for positive indicators
    'warning': '#ca8a04',     # Amber for warnings
    'error': '#dc2626',       # Red for errors
}

# Spacing and Layout
PADDING = {
    'small': 4,
    'medium': 8,
    'large': 12,
    'xlarge': 16,
}

MARGINS = {
    'panel': 8,
    'section': 12,
    'content': 16,
}

# Widget Dimensions
SIZES = {
    'button_height': 28,
    'entry_height': 24,
    'spinbox_width': 80,
    'palette_height': 120,
    'metrics_height': 200,
    'canvas_min_width': 600,
    'sidebar_width': 280,
}

# Border and Relief Styles
BORDER = {
    'width': 1,
    'relief': 'solid',  # Clean flat borders
    'radius': 4,        # For rounded corners if supported
}

# Plot/Chart Styling
PLOT_STYLE = {
    'figure_bg': '#ffffff',
    'axes_bg': '#ffffff',
    'grid_color': '#e5e5e5',
    'text_color': '#2c2c2c',
    'line_width': 1.5,
    'marker_size': 4,
}

# Element Palette Configuration
PALETTE_CONFIG = {
    'columns': 4,
    'button_size': (40, 40),
    'symbol_font_size': 12,
    'name_font_size': 8,
}

# Layout Ratios (for grid-based layouts)
LAYOUT_RATIOS = {
    'canvas_width_ratio': 0.7,    # 70% for canvas
    'sidebar_width_ratio': 0.3,   # 30% for sidebar
    'palette_height_ratio': 0.4,  # 40% of sidebar for palette
    'metrics_height_ratio': 0.6,  # 60% of sidebar for metrics/plots
}