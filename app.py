import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import os

# Page config
st.set_page_config(
    page_title="WMC Drag Reduction Analysis",
    layout="wide"
)

# Clean, professional styling
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #f8f9fa;
        color: #2c3e50;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-left: 4px solid #E02B20
        border: 1px solid #e9ecef;
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    
    .header-title {
        font-size: 2.5em;
        font-weight: 700;
        color: #2c3e50 !important;
        margin: 0;
        text-align: left;
    }
    
    .header-subtitle {
        font-size: 1.2em;
        color: #E02B20 !important;
        margin: 10px 0 0 0;
        font-weight: 500;
        text-align: left;
    }
    
    .wmc-tagline {
        margin-top: 20px;
        font-size: 0.9em;
        color: #7f8c8d !important;
        font-weight: 400;
        letter-spacing: 1px;
        text-transform: uppercase;
        text-align: left;
    }
    
    /* Style the expander to look like the desired container */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        border: 1px solid #e9ecef !important;
        border-left: 3px solid #E02B20 !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 15px !important;
        font-weight: 600 !important;
        color: #2c3e50 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #f8f9fa !important;
        border: 1px solid #e9ecef !important;
        border-top: none !important;
        border-left: 3px solid #E02B20 !important;
        border-radius: 0 0 8px 8px !important;
        padding: 25px !important;
    }
    
    /* Alternative targeting for different Streamlit versions */
    div[data-testid="stExpander"] {
        border: 1px solid #e9ecef !important;
        border-left: 3px solid #E02B20 !important;
        border-radius: 8px !important;
        background-color: #f8f9fa !important;
    }
    
    div[data-testid="stExpander"] > div {
        background-color: #f8f9fa !important;
    }
    
    /* Remove the simple header styling since we're using expander */
    
    /* Input field styling - white backgrounds with black text */
    .stNumberInput label, .stTextInput label, .stSelectbox label, .stRadio label {
        color: #333333 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    /* Target input fields with multiple selectors to ensure styling takes effect */
    .stNumberInput input, 
    .stTextInput input,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input,
    input[type="number"],
    input[type="text"] {
        background-color: white !important;
        color: #333333 !important;
        border: 2px solid #dee2e6 !important;
        border-radius: 6px !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }
    
    /* Focus states */
    .stNumberInput input:focus, 
    .stTextInput input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus,
    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stTextInput"] input:focus,
    input[type="number"]:focus,
    input[type="text"]:focus {
        border-color: #E02B20 !important;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1) !important;
        background-color: white !important;
        color: #333333 !important;
    }
    
    /* Disabled input styling (for CdA display) - comprehensive targeting */
    .stTextInput input:disabled, 
    .stTextInput > div > div > input:disabled,
    div[data-testid="stTextInput"] input:disabled,
    input[type="text"]:disabled,
    input:disabled {
        background-color: #f8f9fa !important;
        color: #333333 !important;
        border: 2px solid #dee2e6 !important;
        opacity: 1 !important;
        font-weight: 600 !important;
        -webkit-text-fill-color: #333333 !important;
        -webkit-opacity: 1 !important;
    }
    
    /* Force text color for disabled inputs - webkit specific */
    input:disabled::-webkit-input-placeholder {
        color: #333333 !important;
    }
    
    /* Ensure input wrapper divs don't interfere */
    .stNumberInput > div > div,
    .stTextInput > div > div {
        background-color: transparent !important;
    }
    
    /* Additional targeting for input containers inside the custom container */
    .input-container input {
        background-color: white !important;
        color: #333333 !important;
    }
    
    .input-container input:disabled {
        background-color: #f8f9fa !important;
        color: #333333 !important;
        -webkit-text-fill-color: #333333 !important;
    }
    
    /* Button styling - consistent grey design with white text */
    .stButton > button {
        background: #6c757d !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
        font-size: 14px !important;
    }
    
    .stButton > button:hover {
        background: #5a6268 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(108, 117, 125, 0.3) !important;
        color: white !important;
    }
    
    .stButton > button:focus {
        color: white !important;
        background: #6c757d !important;
    }
    
    .stButton > button:active {
        color: white !important;
        background: #5a6268 !important;
    }
    
    /* Target button text specifically */
    .stButton button p {
        color: white !important;
    }
    
    .stButton button span {
        color: white !important;
    }
    
    .stButton button div {
        color: white !important;
    }
    
    /* Override Streamlit's default button text color */
    button[data-testid="baseButton-secondary"] {
        background: #6c757d !important;
        color: white !important;
    }
    
    button[data-testid="baseButton-secondary"]:hover {
        background: #5a6268 !important;
        color: white !important;
    }
    
    /* Technical specs styling */
    .technical-specs {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
        font-size: 12px;
        color: #6c757d !important;
        text-align: center;
        border-left: 3px solid #E02B20;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: #f8f9fa !important;
        border: 1px solid #e9ecef !important;
        padding: 15px !important;
        border-radius: 8px !important;
        border-left: 3px solid #E02B20 !important;
    }
    
    .stRadio label {
        color: #2c3e50 !important;
        font-weight: 500 !important;
    }
    
    /* Section headers */
    .stSubheader {
        color: #2c3e50 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #E02B20 !important;
        padding-bottom: 5px !important;
        margin-bottom: 20px !important;
    }
    
    /* Checkbox styling */
    .stCheckbox label {
        color: #2c3e50 !important;
        font-weight: 500 !important;
    }
    
    /* Metric container styling */
    [data-testid="metric-container"] {
        background-color: #f8f9fa !important;
        border: 1px solid #e9ecef !important;
        border-left: 3px solid #E02B20!important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    
    [data-testid="metric-container"] > div {
        color: #2c3e50 !important;
    }
    
    /* Footer styling */
    .footer {
        margin-top: 40px;
        text-align: center;
        padding: 20px;
        border-top: 2px solid #E02B20;
        color: #6c757d !important;
        background: #f8f9fa;
        border-radius: 8px;
    }
    
    .footer a {
        color: #2c3e50 !important;
        text-decoration: none;
        font-weight: 600;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* General text styling */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, span, div {
        color: #2c3e50 !important;
    }
    
    /* Remove any unwanted margins */
    .plot-container {
        margin: 0 !important;
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Constants
CONSTANTS = {
    'CRR': 0.02,           # rolling resistance coefficient
    'g': 9.81,             # gravitational acceleration (m/s²)
    'rho': 1.225,          # air density (kg/m³)
    'efficiency': 1     # drivetrain efficiency
}

# CSV file mapping
CSV_FILES = {
    'WMC_E': 'data/WMC_E.csv',
    'UDDS': 'data/UDDS.csv',
    'WLTP_1': 'data/WLTC_1.csv',
    'WLTP_2': 'data/WLTC_2.csv',
    'WLTP_3': 'data/WLTC_3.csv'
}

# Color palette
COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']

# Initialize session state for preset handling
if 'preset_values' not in st.session_state:
    st.session_state.preset_values = {
        'mass': 150,
        'drag_coeff': 0.5,
        'frontal_area': 1.0,
        'top_speed': 180
    }

@st.cache_data
def load_drive_cycle_data():
    drive_cycles = {}
    missing_files = []
    total_points = 0

    for cycle_name, filename in CSV_FILES.items():
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)

                # Handle the exact column names from your CSV files
                if 'Elapsed Time (s)' in df.columns and 'Speed (kph)' in df.columns:
                    cycle_data = df[['Elapsed Time (s)', 'Speed (kph)']].copy()
                    cycle_data.columns = ['time', 'speed']
                    cycle_data = cycle_data.dropna()

                    drive_cycles[cycle_name] = cycle_data
                    total_points += len(cycle_data)
                else:
                    st.error(f"Expected columns 'Elapsed Time (s)' and 'Speed (kph)' in {filename}")

            except Exception as e:
                st.error(f"Error reading {filename}: {str(e)}")
        else:
            missing_files.append(filename)

    return drive_cycles, missing_files, total_points

def calculate_cda(drag_coeff, frontal_area):
    return drag_coeff * frontal_area

def infer_power(top_speed_kmh, mass, cda):
    v_ms = top_speed_kmh / 3.6
    effective_power = (0.5 * CONSTANTS['rho'] * cda * v_ms**3 +
                      CONSTANTS['CRR'] * mass * CONSTANTS['g'] * v_ms)
    return effective_power / CONSTANTS['efficiency']

def solve_top_speed(power, mass, cda):
    target = power * CONSTANTS['efficiency']
    v = 10.0  # initial guess in m/s

    for _ in range(50):
        f = (0.5 * CONSTANTS['rho'] * cda * v**3 +
             CONSTANTS['CRR'] * mass * CONSTANTS['g'] * v - target)
        df = (1.5 * CONSTANTS['rho'] * cda * v**2 +
              CONSTANTS['CRR'] * mass * CONSTANTS['g'])

        new_v = v - f / df
        if abs(new_v - v) < 1e-6:
            break
        v = new_v

    return v * 3.6  # convert to km/h

def calculate_drag_force(speed_kmh, mass, drag_coeff, frontal_area):
    v_ms = speed_kmh / 3.6
    f_roll = CONSTANTS['CRR'] * mass * CONSTANTS['g']
    cda = calculate_cda(drag_coeff, frontal_area)
    f_aero = 0.5 * CONSTANTS['rho'] * cda * v_ms**2
    f_total = f_roll + f_aero

    return {'rolling': f_roll, 'aerodynamic': f_aero, 'total': f_total}


def calculate_instantaneous_power(speed_kmh, acceleration_ms2, mass, cda):
    """
    Calculate instantaneous mechanical power (no efficiency)
    """
    v_ms = speed_kmh / 3.6

    # Forces
    f_slope = 0  # roadSlope = 0
    f_roll = CONSTANTS['CRR'] * mass * CONSTANTS['g']
    f_aero = 0.5 * CONSTANTS['rho'] * cda * v_ms ** 2
    f_accel = mass * acceleration_ms2

    # Total force
    f_total = f_slope + f_roll + f_aero + f_accel

    # Power (W)
    power = f_total * v_ms

    # Only positive power
    return max(0, power)


def analyze_drive_cycle(cycle_data, mass, cda):
    """
    Analyze drive cycle matching Scilab discrete-time model exactly
    """
    total_energy_j = 0
    total_distance_m = 0
    max_power_w = 0

    speeds_kmh = cycle_data['speed'].values
    times_s = cycle_data['time'].values

    # Initialize previous speed for discrete derivative
    v_prev_ms = 0  # Initial speed is 0

    for i in range(len(cycle_data)):
        # Current speed
        v_current_ms = speeds_kmh[i] / 3.6

        if i == 0:
            # First point - no acceleration
            acceleration = 0
            dt = times_s[0]  # Time from 0 to first point
        else:
            # Time step
            dt = times_s[i] - times_s[i - 1]
            if dt <= 0:
                continue

            # Discrete derivative (matching 1/z and dT blocks in Scilab)
            acceleration = (v_current_ms - v_prev_ms) / dt

        # Calculate forces at current speed
        f_slope = 0  # roadSlope = 0
        f_roll = CONSTANTS['CRR'] * mass * CONSTANTS['g']
        f_aero = 0.5 * CONSTANTS['rho'] * cda * v_current_ms ** 2
        f_accel = mass * acceleration

        # Total resistive force
        f_total = f_slope + f_roll + f_aero + f_accel

        # Power (W)
        power_w = f_total * v_current_ms

        # Only positive power (matching dynamic switch in Scilab)
        if power_w > 0:
            max_power_w = max(max_power_w, power_w)

            # Energy integration (only for positive power)
            if i > 0:  # Don't integrate the first point
                energy_j = power_w * dt
                total_energy_j += energy_j

        # Distance integration
        if i > 0:
            distance_m = v_current_ms * dt
            total_distance_m += distance_m

        # Update previous speed for next iteration
        v_prev_ms = v_current_ms

    # Convert units (matching Scilab's 1/3.6e6 and distance calculation)
    total_energy_kwh = total_energy_j / 3.6e6
    total_distance_km = total_distance_m / 1000

    # Energy per km
    energy_per_km = (total_energy_kwh * 1000) / total_distance_km if total_distance_km > 0 else 0

    # Average power
    total_time_hours = times_s[-1] / 3600
    avg_power_kw = total_energy_kwh / total_time_hours if total_time_hours > 0 else 0

    return {
        'total_energy': total_energy_kwh,
        'total_distance': total_distance_km,
        'energy_per_km': energy_per_km,
        'avg_power': avg_power_kw,
        'max_power': max_power_w / 1000
    }

def generate_drag_analysis(mass, base_cda, frontal_area, max_speed_kmh):
    speeds = list(range(0, int(max_speed_kmh) + 1, 5))
    variations = [-25, -20, -15, -10, -5, 0]
    drag_data = {}

    for percent in variations:
        # Apply variations to CdA directly
        adjusted_cda = base_cda * (1 + percent / 100)
        # Calculate equivalent drag coefficient for the drag force calculation
        adjusted_drag_coeff = adjusted_cda / frontal_area
        forces = [calculate_drag_force(speed, mass, adjusted_drag_coeff, frontal_area) for speed in speeds]

        drag_data[percent] = {
            'drag_coeff': adjusted_drag_coeff,
            'cda': adjusted_cda,
            'total_forces': [f['total'] for f in forces],
            'rolling_forces': [f['rolling'] for f in forces],
            'aero_forces': [f['aerodynamic'] for f in forces]
        }

    # Calculate drag reductions
    baseline_forces = drag_data[0]['total_forces']
    for percent in variations:
        if percent != 0:
            forces = drag_data[percent]['total_forces']
            drag_data[percent]['drag_reductions'] = [baseline - force for baseline, force in zip(baseline_forces, forces)]
        else:
            drag_data[percent]['drag_reductions'] = [0] * len(speeds)

    return {
        'speeds': speeds,
        'variations': variations,
        'drag_data': drag_data
    }

def perform_analysis(mass, base_drag_coeff, frontal_area, current_top_speed, drive_cycles):
    base_cda = calculate_cda(base_drag_coeff, frontal_area)
    inferred_power = infer_power(current_top_speed, mass, base_cda)

    # CdA variations (reductions only) - now applied to CdA directly
    variations = [-25, -20, -15, -10, -5, 0]

    results = {
        'variations': variations,
        'drag_coeffs': [],
        'cdas': [],
        'top_speeds': [],
        'drive_cycle_energy': {},
        'base_power': inferred_power,
        'base_cda': base_cda,
        'drag_analysis': None
    }

    # Analyze top speed variations - apply reductions to CdA directly
    for percent in variations:
        adjusted_cda = base_cda * (1 + percent / 100)
        # Calculate equivalent drag coefficient for display purposes
        adjusted_drag_coeff = adjusted_cda / frontal_area
        top_speed = solve_top_speed(inferred_power, mass, adjusted_cda)

        results['drag_coeffs'].append(adjusted_drag_coeff)
        results['cdas'].append(adjusted_cda)
        results['top_speeds'].append(top_speed)

    # Analyze drive cycles using real data
    for cycle_name, cycle_data in drive_cycles.items():
        if not cycle_data.empty:
            cycle_results = []
            for i, percent in enumerate(variations):
                cda = results['cdas'][i]
                analysis = analyze_drive_cycle(cycle_data, mass, cda)
                analysis['variation'] = percent
                cycle_results.append(analysis)
            results['drive_cycle_energy'][cycle_name] = cycle_results

    # Generate drag analysis - now working with CdA variations
    results['drag_analysis'] = generate_drag_analysis(mass, base_cda, frontal_area, current_top_speed)

    return results

def get_plot_layout(title, xaxis_title, yaxis_title):
    """Get consistent plot layout with white background and clean styling"""
    return {
        'title': {
            'text': title,
            'font': {'size': 18, 'color': '#2c3e50', 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        'xaxis': {
            'title': {'text': xaxis_title, 'font': {'size': 14, 'color': '#2c3e50'}},
            'gridcolor': '#e9ecef',
            'linecolor': '#dee2e6',
            'tickfont': {'color': '#2c3e50'}
        },
        'yaxis': {
            'title': {'text': yaxis_title, 'font': {'size': 14, 'color': '#2c3e50'}},
            'gridcolor': '#e9ecef',
            'linecolor': '#dee2e6',
            'tickfont': {'color': '#2c3e50'}
        },
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'height': 400,
        'margin': dict(t=60, b=60, l=60, r=60),
        'font': {'color': '#2c3e50'},
        'legend': {
            'font': {'color': '#2c3e50'},
            'bgcolor': 'rgba(255, 255, 255, 0.8)',
            'bordercolor': '#dee2e6',
            'borderwidth': 1
        }
    }

def create_plots(results, is_percentage_mode):
    reduction_percentages = [abs(v) for v in results['variations']]

    if not is_percentage_mode:
        # Absolute mode plots

        # Plot 1: Top Speed vs CdA Reduction
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=reduction_percentages,
            y=results['top_speeds'],
            mode='lines+markers',
            name='Top Speed',
            line=dict(color=COLORS[0], width=4),
            marker=dict(size=10, color=COLORS[0])
        ))
        fig1.update_layout(**get_plot_layout(
            'Top Speed vs CdA Reduction',
            'CdA Reduction (%)',
            'Top Speed (km/h)'
        ))
        fig1.update_layout(showlegend=False)

        # Plot 2: Energy Consumption per km
        fig2 = go.Figure()
        for i, (cycle, data) in enumerate(results['drive_cycle_energy'].items()):
            energy_per_km = [d['energy_per_km'] for d in data]
            fig2.add_trace(go.Scatter(
                x=reduction_percentages,
                y=energy_per_km,
                mode='lines+markers',
                name=cycle.replace('_', ' '),
                line=dict(color=COLORS[i % len(COLORS)], width=3),
                marker=dict(size=8, color=COLORS[i % len(COLORS)])
            ))
        fig2.update_layout(**get_plot_layout(
            'Energy Consumption per km vs CdA Reduction',
            'CdA Reduction (%)',
            'Energy Consumption (Wh/km)'
        ))

        # Plot 3: Total Drag Force vs Speed
        fig3 = go.Figure()
        for i, percent in enumerate(results['drag_analysis']['variations']):
            data = results['drag_analysis']['drag_data'][percent]
            reduction_percent = abs(percent)
            label = f'CdA -{reduction_percent}%' if percent < 0 else 'Baseline'

            fig3.add_trace(go.Scatter(
                x=results['drag_analysis']['speeds'],
                y=data['total_forces'],
                mode='lines',
                name=label,
                line=dict(
                    color=COLORS[i % len(COLORS)],
                    width=4 if percent == 0 else 3,
                    dash='solid' if percent == 0 else 'dash'
                )
            ))
        fig3.update_layout(**get_plot_layout(
            'Aerodynamic Drag vs Vehicle Speed',
            'Vehicle Speed (km/h)',
            'Drag Force (N)'
        ))

        # Plot 4: Drag Reduction vs Speed
        fig4 = go.Figure()
        for i, percent in enumerate(results['drag_analysis']['variations']):
            if percent < 0:
                data = results['drag_analysis']['drag_data'][percent]
                reduction_percent = abs(percent)

                fig4.add_trace(go.Scatter(
                    x=results['drag_analysis']['speeds'],
                    y=data['drag_reductions'],
                    mode='lines',
                    name=f'CdA -{reduction_percent}%',
                    line=dict(color=COLORS[i % len(COLORS)], width=3)
                ))
        fig4.update_layout(**get_plot_layout(
            'Drag Force Reduction vs Vehicle Speed',
            'Vehicle Speed (km/h)',
            'Drag Reduction (N)'
        ))

    else:
        # Percentage mode plots

        # Plot 1: Top Speed Change (%)
        baseline_speed = results['top_speeds'][results['variations'].index(0)]
        speed_changes = [(speed - baseline_speed) / baseline_speed * 100 for speed in results['top_speeds']]

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=reduction_percentages,
            y=speed_changes,
            mode='lines+markers',
            name='Top Speed Change',
            line=dict(color=COLORS[1], width=4),
            marker=dict(size=10, color=COLORS[1])
        ))
        fig1.update_layout(**get_plot_layout(
            'Top Speed vs CdA Reduction',
            'CdA Reduction (%)',
            'Top Speed Change (%)'
        ))
        fig1.update_layout(showlegend=False)

        # Plot 2: Energy Consumption Change (%)
        fig2 = go.Figure()
        for i, (cycle, data) in enumerate(results['drive_cycle_energy'].items()):
            baseline_energy = data[results['variations'].index(0)]['energy_per_km']
            energy_changes = [(d['energy_per_km'] - baseline_energy) / baseline_energy * 100 for d in data]

            fig2.add_trace(go.Scatter(
                x=reduction_percentages,
                y=energy_changes,
                mode='lines+markers',
                name=cycle.replace('_', ' '),
                line=dict(color=COLORS[i % len(COLORS)], width=3),
                marker=dict(size=8, color=COLORS[i % len(COLORS)])
            ))
        fig2.update_layout(**get_plot_layout(
            'Energy Consumption per km vs CdA Reduction',
            'CdA Reduction (%)',
            'Energy Consumption Change (%)'
        ))

        # Plot 3: Total Drag Force Change (%)
        fig3 = go.Figure()
        baseline_forces = results['drag_analysis']['drag_data'][0]['total_forces']

        for i, percent in enumerate(results['drag_analysis']['variations']):
            if percent < 0:
                data = results['drag_analysis']['drag_data'][percent]
                reduction_percent = abs(percent)
                force_changes = [(force - baseline) / baseline * 100 for force, baseline in zip(data['total_forces'], baseline_forces)]

                fig3.add_trace(go.Scatter(
                    x=results['drag_analysis']['speeds'],
                    y=force_changes,
                    mode='lines',
                    name=f'CdA -{reduction_percent}%',
                    line=dict(color=COLORS[i % len(COLORS)], width=3)
                ))
        fig3.update_layout(**get_plot_layout(
            'Total Drag Force vs Vehicle Speed',
            'Vehicle Speed (km/h)',
            'Total Drag Force Change (%)'
        ))

        # Plot 4: Drag Reduction as % of Baseline
        fig4 = go.Figure()
        for i, percent in enumerate(results['drag_analysis']['variations']):
            if percent < 0:
                data = results['drag_analysis']['drag_data'][percent]
                reduction_percent = abs(percent)
                reduction_percentages_data = [(reduction / baseline) * 100 for reduction, baseline in zip(data['drag_reductions'], baseline_forces)]

                fig4.add_trace(go.Scatter(
                    x=results['drag_analysis']['speeds'],
                    y=reduction_percentages_data,
                    mode='lines',
                    name=f'CdA -{reduction_percent}%',
                    line=dict(color=COLORS[i % len(COLORS)], width=3)
                ))
        fig4.update_layout(**get_plot_layout(
            'Drag Force Reduction vs Vehicle Speed',
            'Vehicle Speed (km/h)',
            'Drag Reduction (% of Baseline)'
        ))

    return fig1, fig2, fig3, fig4

def create_drive_cycle_plot(drive_cycles, selected_cycles):
    fig = go.Figure()

    for i, cycle_name in enumerate(selected_cycles):
        if cycle_name in drive_cycles:
            cycle_data = drive_cycles[cycle_name]
            fig.add_trace(go.Scatter(
                x=cycle_data['time'],
                y=cycle_data['speed'],
                mode='lines',
                name=cycle_name.replace('_', ' '),
                line=dict(color=COLORS[i % len(COLORS)], width=3)
            ))

    fig.update_layout(**get_plot_layout(
        'Drive Cycle Data',
        'Time (seconds)',
        'Speed (km/h)'
    ))

    return fig

# Main application
def main():
    # Header with clean branding
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">Vehicle Energy Analysis Tool</h1>
        <p class="header-subtitle">WHITE MOTORCYCLE CONCEPTS</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    drive_cycles, missing_files, total_points = load_drive_cycle_data()

    if missing_files:
        st.error(f"Missing CSV files: {', '.join(missing_files)}")
        st.info("Please place the CSV files in the same directory as this script.")
        if not drive_cycles:
            st.stop()

    # Preset buttons
    st.subheader("Vehicle Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("1000cc 4 Cylinder Superbike", use_container_width=True):
            st.session_state.preset_values = {
                'mass': 201,
                'drag_coeff': 0.42,
                'frontal_area': 1.0,
                'top_speed': 302
            }
            st.rerun()

    with col2:
        if st.button("700cc Parallel Twin Sports Tourer", use_container_width=True):
            st.session_state.preset_values = {
                'mass': 201,
                'drag_coeff': 0.65,
                'frontal_area': 1.0,
                'top_speed': 201
            }
            st.rerun()

    with col3:
        if st.button("250cc Single Cylinder Sports Bike ", use_container_width=True):
            st.session_state.preset_values = {
                'mass': 161,
                'drag_coeff': 0.54,
                'frontal_area': 1.0,
                'top_speed': 150
            }
            st.rerun()

    # Input section with clean styling
    with st.expander("", expanded=True):

        # Vehicle parameters row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            mass = st.number_input(
                "Vehicle Mass (kg)",
                min_value=100, max_value=5000,
                value=st.session_state.preset_values['mass'],
                step=10
            )

        with col2:
            drag_coeff = st.number_input(
                "Drag Coefficient (Cd)",
                min_value=0.1, max_value=1.0,
                value=st.session_state.preset_values['drag_coeff'],
                step=0.01,
                format="%.2f"
            )

        with col3:
            frontal_area = st.number_input(
                "Frontal Area (m²)",
                min_value=1.0, max_value=5.0,
                value=st.session_state.preset_values['frontal_area'],
                step=0.1,
                format="%.1f"
            )

        with col4:
            top_speed = st.number_input(
                "Current Top Speed (km/h)",
                min_value=50, max_value=400,
                value=st.session_state.preset_values['top_speed'],
                step=5
            )

        with col5:
            cda = calculate_cda(drag_coeff, frontal_area)
            st.text_input("CdA", value=f"{cda:.3f} m²", disabled=True)

        # Controls row
        st.markdown("")
        control_col1, control_col2 = st.columns([2, 1])

        with control_col1:
            display_mode = st.radio("Display Mode", ["Absolute Values", "Percentage Values"], horizontal=True)
            is_percentage_mode = display_mode == "Percentage Values"

        with control_col2:
            if st.button("RUN ANALYSIS", use_container_width=True):
                analyze_button = True
            else:
                analyze_button = False

    # Update preset values when inputs change
    st.session_state.preset_values = {
        'mass': mass,
        'drag_coeff': drag_coeff,
        'frontal_area': frontal_area,
        'top_speed': top_speed
    }

    # Technical specs
    cycle_info = ", ".join([f"{name} ({len(data)} pts)" for name, data in drive_cycles.items()])
    st.markdown(f"""
    <div class="technical-specs">
        <strong>Analysis Constants:</strong> Rolling Resistance = 0.02 • Air Density = 1.225 kg/m³<br>
    </div>
    """, unsafe_allow_html=True)

    # Analysis results
    if analyze_button or 'analysis_results' in st.session_state:
        if analyze_button:
            with st.spinner("Running analysis..."):
                results = perform_analysis(mass, drag_coeff, frontal_area, top_speed, drive_cycles)
                st.session_state['analysis_results'] = results
        else:
            results = st.session_state['analysis_results']

        # Create 2x2 plot grid
        fig1, fig2, fig3, fig4 = create_plots(results, is_percentage_mode)

        plot_col1, plot_col2 = st.columns(2)

        with plot_col1:
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig3, use_container_width=True)

        with plot_col2:
            st.plotly_chart(fig2, use_container_width=True)
            st.plotly_chart(fig4, use_container_width=True)

        # Footer
        st.markdown("""
        <div class="footer">
            <p>INVENT • INNOVATE • CREATE</p>
        </div>
        """, unsafe_allow_html=True)

    # Drive cycle section
    st.header("Drive Cycle Data")

    if st.button("DISPLAY DRIVE CYCLE DATA", use_container_width=True):
        st.session_state['show_drive_cycles'] = not st.session_state.get('show_drive_cycles', False)

    if st.session_state.get('show_drive_cycles', False):
        st.subheader("Select Drive Cycles to Display:")

        cycle_cols = st.columns(5)
        selected_cycles = []

        cycle_names = ['WLTP_1', 'WLTP_2', 'WLTP_3', 'UDDS', 'WMC_E']
        cycle_labels = ['WLTP_1', 'WLTP_2', 'WLTP_3', 'UDDS', 'WMC_E']

        for i, (cycle_name, label) in enumerate(zip(cycle_names, cycle_labels)):
            with cycle_cols[i]:
                if st.checkbox(label, value=True, key=f"cycle_{cycle_name}"):
                    selected_cycles.append(cycle_name)

        # Drive cycle plot
        if selected_cycles:
            fig_cycles = create_drive_cycle_plot(drive_cycles, selected_cycles)
            st.plotly_chart(fig_cycles, use_container_width=True)
        else:
            st.info("No drive cycles selected")

if __name__ == "__main__":
    main()
