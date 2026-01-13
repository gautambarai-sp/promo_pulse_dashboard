"""
Promo Pulse Dashboard - Main Streamlit Application
UAE Retail Analytics & Promotion Simulator
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import io

# Import custom modules
from data_generator import generate_all_data
from cleaner import DataCleaner, clean_uploaded_data
from simulator import KPICalculator, PromoSimulator, format_number_short, format_percentage

# ============== PAGE CONFIGURATION ==============
st.set_page_config(
    page_title="Promo Pulse Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== CUSTOM CSS ==============
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    
    /* KPI Cards */
    .kpi-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #4361ee;
        margin-bottom: 1rem;
        min-height: 120px;
    }
    
    .kpi-title {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        line-height: 1.2;
    }
    
    .kpi-reference {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-top: 0.5rem;
    }
    
    /* Status indicators */
    .status-good { color: #10b981; }
    .status-warning { color: #f59e0b; }
    .status-danger { color: #ef4444; }
    
    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1a1a2e;
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #4361ee;
    }
    
    /* Chart section spacing */
    .chart-section {
        margin: 2rem 0;
        padding: 1.5rem;
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Recommendation box */
    .recommendation-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #0ea5e9;
        margin: 1.5rem 0;
    }
    
    /* Data comparison styling */
    .comparison-container {
        display: flex;
        gap: 1rem;
    }
    
    .comparison-panel {
        flex: 1;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .raw-panel {
        background: #fef2f2;
        border: 2px solid #fecaca;
    }
    
    .clean-panel {
        background: #f0fdf4;
        border: 2px solid #bbf7d0;
    }
    
    /* Toggle styling */
    .view-toggle {
        background: #f3f4f6;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 1rem;
    }
    
    /* Error log styling */
    .error-item {
        background: #fef2f2;
        border-left: 3px solid #ef4444;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 4px 4px 0;
        font-size: 0.85rem;
    }
    
    .warning-item {
        background: #fffbeb;
        border-left: 3px solid #f59e0b;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 4px 4px 0;
        font-size: 0.85rem;
    }
    
    .success-item {
        background: #f0fdf4;
        border-left: 3px solid #10b981;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 4px 4px 0;
        font-size: 0.85rem;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Dashboard intro */
    .dashboard-intro {
        background: linear-gradient(135deg, #4361ee 0%, #3730a3 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
    }
    
    .dashboard-intro h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .dashboard-intro p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }
    
    /* Full width chart container */
    .full-width-chart {
        width: 100%;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============== SESSION STATE INITIALIZATION ==============
def init_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'use_clean_data' not in st.session_state:
        st.session_state.use_clean_data = True
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'Manager'
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = {}
    if 'clean_data' not in st.session_state:
        st.session_state.clean_data = {}
    if 'issues_log' not in st.session_state:
        st.session_state.issues_log = pd.DataFrame()
    if 'cleaning_summary' not in st.session_state:
        st.session_state.cleaning_summary = {}
    if 'use_uploaded_data' not in st.session_state:
        st.session_state.use_uploaded_data = False
    if 'filters' not in st.session_state:
        st.session_state.filters = {}


def load_or_generate_data():
    """Load existing data or generate new synthetic data"""
    raw_dir = 'data/raw'
    clean_dir = 'data/clean'
    
    # Check if data exists
    if not os.path.exists(f'{raw_dir}/sales_raw.csv'):
        with st.spinner('Generating synthetic data with intentional errors...'):
            generate_all_data(raw_dir)
    
    # Check if clean data exists
    if not os.path.exists(f'{clean_dir}/sales_clean.csv'):
        with st.spinner('Cleaning data and generating error logs...'):
            cleaner = DataCleaner(clean_dir)
            result = cleaner.clean_all_data(raw_dir)
            st.session_state.cleaning_summary = result.get('summary', {})
    
    # Load raw data
    st.session_state.raw_data = {
        'products': pd.read_csv(f'{raw_dir}/products_raw.csv'),
        'stores': pd.read_csv(f'{raw_dir}/stores_raw.csv'),
        'customers': pd.read_csv(f'{raw_dir}/customers_raw.csv'),
        'sales': pd.read_csv(f'{raw_dir}/sales_raw.csv'),
        'inventory': pd.read_csv(f'{raw_dir}/inventory_raw.csv'),
        'campaigns': pd.read_csv(f'{raw_dir}/campaigns_raw.csv'),
        'departments': pd.read_csv(f'{raw_dir}/departments.csv')
    }
    
    # Load clean data
    st.session_state.clean_data = {
        'products': pd.read_csv(f'{clean_dir}/products_clean.csv'),
        'stores': pd.read_csv(f'{clean_dir}/stores_clean.csv'),
        'customers': pd.read_csv(f'{clean_dir}/customers_clean.csv'),
        'sales': pd.read_csv(f'{clean_dir}/sales_clean.csv'),
        'inventory': pd.read_csv(f'{clean_dir}/inventory_clean.csv'),
        'campaigns': pd.read_csv(f'{clean_dir}/campaigns_clean.csv')
    }
    
    # Load issues log
    if os.path.exists(f'{clean_dir}/logs/issues_log.csv'):
        st.session_state.issues_log = pd.read_csv(f'{clean_dir}/logs/issues_log.csv')
    
    st.session_state.data_loaded = True


def render_kpi_card(title: str, value, reference: str = None, status: str = None, delta: str = None):
    """Render a KPI card with optional reference and status"""
    status_class = f"status-{status}" if status else ""
    delta_html = f'<div class="{status_class}" style="font-size: 0.9rem; margin-top: 0.3rem;">{delta}</div>' if delta else ""
    reference_html = f'<div class="kpi-reference">{reference}</div>' if reference else ""
    
    # Ensure value is a string
    value_str = str(value) if value is not None else "N/A"
    
    st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value {status_class}">{value_str}</div>
            {delta_html}
            {reference_html}
        </div>
    """, unsafe_allow_html=True)


def get_data():
    """Get currently selected data (raw or clean, uploaded or pre-built)"""
    if st.session_state.use_uploaded_data and st.session_state.uploaded_data:
        return st.session_state.uploaded_data
    elif st.session_state.use_clean_data:
        return st.session_state.clean_data
    return st.session_state.raw_data


# ============== SIDEBAR ==============
def render_sidebar():
    """Render the sidebar with filters and controls"""
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Data source toggle
        st.markdown("### 📁 Data Source")
        data_source = st.radio(
            "Select Dataset",
            ["Pre-built (Synthetic)", "Upload Custom"],
            key="data_source_radio"
        )
        
        if data_source == "Upload Custom":
            render_upload_section()
            st.session_state.use_uploaded_data = True
        else:
            st.session_state.use_uploaded_data = False
        
        st.markdown("---")
        
        # Data toggle (raw vs clean)
        st.markdown("### 🔄 Data Type")
        use_clean = st.toggle("Use Cleaned Data", value=True, key="clean_toggle")
        st.session_state.use_clean_data = use_clean
        
        if use_clean:
            st.success("✓ Using cleaned data")
        else:
            st.warning("⚠ Using raw data (may contain errors)")
        
        st.markdown("---")
        
        # View toggle
        st.markdown("### 👁️ Dashboard View")
        view = st.radio(
            "Select View",
            ["Manager", "Executive"],
            key="view_toggle"
        )
        st.session_state.current_view = view
        
        st.markdown("---")
        
        # Filters
        st.markdown("### 🔍 Filters")
        
        data = get_data()
        
        if data and 'sales' in data and len(data['sales']) > 0:
            # Get unique values for filters
            if 'stores' in data:
                stores_df = data['stores']
                cities = ['All'] + sorted([c for c in stores_df['city'].unique() if pd.notna(c)])
                channels = ['All'] + sorted([c for c in stores_df['channel'].unique() if pd.notna(c)])
            else:
                cities = ['All']
                channels = ['All']
            
            if 'products' in data:
                products_df = data['products']
                categories = ['All'] + sorted([c for c in products_df['category'].unique() if pd.notna(c)])
                brands = ['All'] + sorted([b for b in products_df['brand'].unique() if pd.notna(b)])[:20]  # Limit brands
            else:
                categories = ['All']
                brands = ['All']
            
            selected_city = st.selectbox("City", cities, key="filter_city")
            selected_channel = st.selectbox("Channel", channels, key="filter_channel")
            selected_category = st.selectbox("Category", categories, key="filter_category")
            selected_brand = st.selectbox("Brand", brands, key="filter_brand")
            
            # Date range (for 2024 data)
            st.markdown("#### Date Range")
            date_start = st.date_input("From", datetime(2024, 1, 1), key="filter_date_start")
            date_end = st.date_input("To", datetime(2024, 12, 31), key="filter_date_end")
            
            # Store filters in session state
            st.session_state.filters = {
                'city': selected_city if selected_city != 'All' else None,
                'channel': selected_channel if selected_channel != 'All' else None,
                'category': selected_category if selected_category != 'All' else None,
                'brand': selected_brand if selected_brand != 'All' else None,
                'date_start': str(date_start),
                'date_end': str(date_end)
            }
        else:
            st.session_state.filters = {}
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Regenerate Data", key="regen_btn"):
            with st.spinner("Regenerating data..."):
                generate_all_data('data/raw')
                cleaner = DataCleaner('data/clean')
                cleaner.clean_all_data('data/raw')
                st.session_state.data_loaded = False
                st.rerun()
        
        if st.button("📥 Download Issues Log", key="download_issues"):
            if len(st.session_state.issues_log) > 0:
                csv = st.session_state.issues_log.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "issues_log.csv",
                    "text/csv"
                )


def render_upload_section():
    """Render file upload section for custom datasets"""
    st.markdown("#### Upload Datasets")
    
    uploaded_files = st.file_uploader(
        "Upload CSV/Excel files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        key="file_uploader",
        help="Upload your sales, products, stores, and inventory files"
    )
    
    if uploaded_files:
        uploaded_data = {}
        for file in uploaded_files:
            try:
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                
                # Detect file type from name
                file_key = file.name.split('.')[0].lower()
                
                # Map common file names to expected keys
                key_mapping = {
                    'sales': 'sales', 'sales_raw': 'sales', 'sales_clean': 'sales',
                    'products': 'products', 'products_raw': 'products', 'products_clean': 'products',
                    'stores': 'stores', 'stores_raw': 'stores', 'stores_clean': 'stores',
                    'inventory': 'inventory', 'inventory_raw': 'inventory', 'inventory_clean': 'inventory',
                    'customers': 'customers', 'customers_raw': 'customers', 'customers_clean': 'customers',
                    'campaigns': 'campaigns', 'campaigns_raw': 'campaigns', 'campaigns_clean': 'campaigns'
                }
                
                mapped_key = key_mapping.get(file_key, file_key)
                uploaded_data[mapped_key] = df
                st.success(f"✓ Loaded {file.name} ({len(df):,} rows)")
            except Exception as e:
                st.error(f"Error loading {file.name}: {str(e)}")
        
        if uploaded_data:
            # Only store raw data if not already cleaned
            if 'uploaded_data' not in st.session_state or st.session_state.uploaded_data is None:
                st.session_state.uploaded_data = uploaded_data
            
            # Clean the uploaded data
            if st.button("🧹 Clean Uploaded Data", key="clean_uploaded_btn"):
                with st.spinner("Cleaning uploaded data..."):
                    try:
                        cleaned_data = clean_uploaded_data(uploaded_data)
                        st.session_state.uploaded_data = cleaned_data
                        issues = cleaned_data.get('issues', pd.DataFrame())
                        if len(issues) > 0:
                            st.session_state.issues_log = issues
                        st.success(f"✓ Data cleaned! {len(issues)} issues found and fixed.")
                        st.session_state.use_uploaded_data = True
                    except Exception as e:
                        st.error(f"Cleaning error: {str(e)}")
            
            # Show column info
            with st.expander("📋 Detected Columns"):
                for table_name, df in uploaded_data.items():
                    st.markdown(f"**{table_name}:** {', '.join(df.columns[:10])}")
                    if len(df.columns) > 10:
                        st.caption(f"... and {len(df.columns) - 10} more columns")


# ============== VISUALIZATION FUNCTIONS ==============
def create_bcg_matrix(data: Dict) -> go.Figure:
    """Create BCG Matrix for channel/category analysis"""
    sales_df = data['sales'].merge(
        data['products'][['product_id', 'category', 'base_price_aed', 'unit_cost_aed']],
        on='product_id', how='left'
    ).merge(
        data['stores'][['store_id', 'city', 'channel']],
        on='store_id', how='left'
    )
    
    # Calculate metrics by channel
    paid_sales = sales_df[sales_df['payment_status'] == 'Paid'] if 'payment_status' in sales_df.columns else sales_df
    
    channel_metrics = paid_sales.groupby('channel').agg({
        'selling_price_aed': 'sum',
        'qty': 'sum',
        'order_id': 'nunique'
    }).reset_index()
    
    channel_metrics.columns = ['channel', 'revenue', 'units_sold', 'orders']
    
    # Calculate market share and growth (simulated)
    total_revenue = channel_metrics['revenue'].sum()
    channel_metrics['market_share'] = channel_metrics['revenue'] / total_revenue * 100
    
    # Simulate growth rate based on order patterns
    np.random.seed(42)
    channel_metrics['growth_rate'] = [15 + np.random.randn() * 5, 
                                       25 + np.random.randn() * 5, 
                                       8 + np.random.randn() * 3][:len(channel_metrics)]
    
    # Create BCG matrix
    fig = go.Figure()
    
    # Add quadrant backgrounds
    fig.add_shape(type="rect", x0=0, y0=10, x1=50, y1=30,
                  fillcolor="rgba(16, 185, 129, 0.1)", line_width=0)
    fig.add_shape(type="rect", x0=50, y0=10, x1=100, y1=30,
                  fillcolor="rgba(59, 130, 246, 0.1)", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=-10, x1=50, y1=10,
                  fillcolor="rgba(156, 163, 175, 0.1)", line_width=0)
    fig.add_shape(type="rect", x0=50, y0=-10, x1=100, y1=10,
                  fillcolor="rgba(245, 158, 11, 0.1)", line_width=0)
    
    # Add quadrant labels
    fig.add_annotation(x=25, y=25, text="Question Marks", showarrow=False,
                      font=dict(size=14, color="#6b7280"))
    fig.add_annotation(x=75, y=25, text="Stars ⭐", showarrow=False,
                      font=dict(size=14, color="#6b7280"))
    fig.add_annotation(x=25, y=0, text="Dogs", showarrow=False,
                      font=dict(size=14, color="#6b7280"))
    fig.add_annotation(x=75, y=0, text="Cash Cows 🐄", showarrow=False,
                      font=dict(size=14, color="#6b7280"))
    
    # Add data points
    colors = {'App': '#4361ee', 'Web': '#f72585', 'Marketplace': '#4cc9f0'}
    
    for _, row in channel_metrics.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['market_share']],
            y=[row['growth_rate']],
            mode='markers+text',
            marker=dict(
                size=row['revenue'] / total_revenue * 150,
                color=colors.get(row['channel'], '#6b7280'),
                opacity=0.7
            ),
            text=row['channel'],
            textposition='top center',
            name=row['channel'],
            hovertemplate=f"<b>{row['channel']}</b><br>" +
                         f"Revenue: AED {row['revenue']:,.0f}<br>" +
                         f"Market Share: {row['market_share']:.1f}%<br>" +
                         f"Growth Rate: {row['growth_rate']:.1f}%<extra></extra>"
        ))
    
    fig.update_layout(
        title=dict(text="BCG Matrix - Channel Performance", font=dict(size=18)),
        xaxis_title="Relative Market Share (%)",
        yaxis_title="Market Growth Rate (%)",
        showlegend=True,
        height=500,
        xaxis=dict(range=[0, 100], dtick=25),
        yaxis=dict(range=[-10, 35], dtick=10),
        plot_bgcolor='white'
    )
    
    return fig


def create_sunburst_chart(data: Dict) -> go.Figure:
    """Create sunburst chart for hierarchical revenue breakdown"""
    sales_df = data['sales'].merge(
        data['products'][['product_id', 'category']],
        on='product_id', how='left'
    ).merge(
        data['stores'][['store_id', 'city', 'channel']],
        on='store_id', how='left'
    )
    
    paid_sales = sales_df[sales_df['payment_status'] == 'Paid'] if 'payment_status' in sales_df.columns else sales_df
    
    # Aggregate by hierarchy
    hierarchy_data = paid_sales.groupby(['city', 'channel', 'category']).agg({
        'selling_price_aed': 'sum'
    }).reset_index()
    hierarchy_data.columns = ['city', 'channel', 'category', 'revenue']
    
    fig = px.sunburst(
        hierarchy_data,
        path=['city', 'channel', 'category'],
        values='revenue',
        color='revenue',
        color_continuous_scale='Blues',
        title='Revenue Hierarchy (City → Channel → Category)'
    )
    
    fig.update_layout(height=550)
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Revenue: AED %{value:,.0f}<extra></extra>'
    )
    
    return fig


def create_heatmap(data: Dict) -> go.Figure:
    """Create heatmap of category performance by channel"""
    sales_df = data['sales'].merge(
        data['products'][['product_id', 'category']],
        on='product_id', how='left'
    ).merge(
        data['stores'][['store_id', 'channel']],
        on='store_id', how='left'
    )
    
    paid_sales = sales_df[sales_df['payment_status'] == 'Paid'] if 'payment_status' in sales_df.columns else sales_df
    
    # Create pivot table
    pivot = paid_sales.pivot_table(
        values='selling_price_aed',
        index='category',
        columns='channel',
        aggfunc='sum',
        fill_value=0
    )
    
    # Format values for display
    text_matrix = pivot.apply(lambda x: x.apply(lambda v: format_number_short(v)))
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        text=text_matrix.values,
        texttemplate="%{text}",
        colorscale='RdYlGn',
        hovertemplate='<b>%{y}</b> - %{x}<br>Revenue: AED %{z:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='Revenue Heatmap (Category × Channel)', font=dict(size=18)),
        xaxis_title='Channel',
        yaxis_title='Category',
        height=450
    )
    
    return fig


def create_revenue_trend(data: Dict, filters: Dict = None) -> go.Figure:
    """Create revenue trend chart"""
    sales_df = data['sales'].copy()
    
    # Parse dates
    if 'order_time' in sales_df.columns:
        sales_df['date'] = pd.to_datetime(sales_df['order_time'], errors='coerce').dt.date
    
    # Apply filters
    if filters and 'date_start' in filters and filters['date_start']:
        sales_df = sales_df[sales_df['date'] >= pd.to_datetime(filters['date_start']).date()]
    if filters and 'date_end' in filters and filters['date_end']:
        sales_df = sales_df[sales_df['date'] <= pd.to_datetime(filters['date_end']).date()]
    
    # Only paid orders
    paid_sales = sales_df[sales_df['payment_status'] == 'Paid'] if 'payment_status' in sales_df.columns else sales_df
    
    # Aggregate by date
    daily_revenue = paid_sales.groupby('date').agg({
        'selling_price_aed': 'sum',
        'qty': 'sum',
        'order_id': 'nunique'
    }).reset_index()
    daily_revenue.columns = ['date', 'revenue', 'units', 'orders']
    
    # Calculate 7-day moving average
    daily_revenue['revenue_ma7'] = daily_revenue['revenue'].rolling(7).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_revenue['date'],
        y=daily_revenue['revenue'],
        mode='lines',
        name='Daily Revenue',
        line=dict(color='rgba(67, 97, 238, 0.3)', width=1),
        hovertemplate='Date: %{x}<br>Revenue: AED %{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_revenue['date'],
        y=daily_revenue['revenue_ma7'],
        mode='lines',
        name='7-Day Moving Avg',
        line=dict(color='#4361ee', width=3),
        hovertemplate='Date: %{x}<br>7-Day Avg: AED %{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='Daily Revenue Trend', font=dict(size=18)),
        xaxis_title='Date',
        yaxis_title='Revenue (AED)',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        hovermode='x unified'
    )
    
    return fig


def create_category_margin_chart(data: Dict) -> go.Figure:
    """Create margin by category chart"""
    sales_df = data['sales'].merge(
        data['products'][['product_id', 'category', 'unit_cost_aed']],
        on='product_id', how='left'
    )
    
    paid_sales = sales_df[sales_df['payment_status'] == 'Paid'] if 'payment_status' in sales_df.columns else sales_df
    
    # Calculate margins by category
    paid_sales['revenue'] = paid_sales['selling_price_aed'] * paid_sales['qty']
    paid_sales['cogs'] = paid_sales['unit_cost_aed'] * paid_sales['qty']
    paid_sales['margin'] = paid_sales['revenue'] - paid_sales['cogs']
    
    category_margins = paid_sales.groupby('category').agg({
        'revenue': 'sum',
        'margin': 'sum'
    }).reset_index()
    
    category_margins['margin_pct'] = (category_margins['margin'] / category_margins['revenue'] * 100).round(1)
    category_margins = category_margins.sort_values('margin_pct', ascending=True)
    
    colors = ['#ef4444' if m < 20 else '#f59e0b' if m < 30 else '#10b981' 
              for m in category_margins['margin_pct']]
    
    fig = go.Figure(go.Bar(
        x=category_margins['margin_pct'],
        y=category_margins['category'],
        orientation='h',
        marker_color=colors,
        text=category_margins['margin_pct'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Margin: %{x:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='Gross Margin by Category', font=dict(size=18)),
        xaxis_title='Gross Margin %',
        yaxis_title='',
        height=400,
        showlegend=False
    )
    
    # Add target line
    fig.add_vline(x=25, line_dash="dash", line_color="#6b7280", 
                  annotation_text="Target: 25%", annotation_position="top")
    
    return fig


def create_stockout_risk_chart(data: Dict) -> go.Figure:
    """Create stockout risk visualization"""
    # Get latest inventory
    inventory_df = data['inventory'].copy()
    inventory_df['snapshot_date'] = pd.to_datetime(inventory_df['snapshot_date'])
    latest_inv = inventory_df.sort_values('snapshot_date').groupby(['product_id', 'store_id']).last().reset_index()
    
    # Merge with store info
    latest_inv = latest_inv.merge(
        data['stores'][['store_id', 'city', 'channel']],
        on='store_id', how='left'
    )
    
    # Calculate stockout risk
    latest_inv['at_risk'] = latest_inv['stock_on_hand'] <= latest_inv['reorder_point']
    
    # Aggregate by city and channel
    risk_by_city = latest_inv.groupby('city').agg({
        'at_risk': ['sum', 'count']
    }).reset_index()
    risk_by_city.columns = ['city', 'at_risk_count', 'total']
    risk_by_city['risk_pct'] = (risk_by_city['at_risk_count'] / risk_by_city['total'] * 100).round(1)
    
    risk_by_channel = latest_inv.groupby('channel').agg({
        'at_risk': ['sum', 'count']
    }).reset_index()
    risk_by_channel.columns = ['channel', 'at_risk_count', 'total']
    risk_by_channel['risk_pct'] = (risk_by_channel['at_risk_count'] / risk_by_channel['total'] * 100).round(1)
    
    # Create subplot
    fig = make_subplots(rows=1, cols=2, subplot_titles=('By City', 'By Channel'))
    
    colors_city = ['#ef4444' if r > 15 else '#f59e0b' if r > 10 else '#10b981' 
                   for r in risk_by_city['risk_pct']]
    colors_channel = ['#ef4444' if r > 15 else '#f59e0b' if r > 10 else '#10b981' 
                      for r in risk_by_channel['risk_pct']]
    
    fig.add_trace(go.Bar(
        x=risk_by_city['city'],
        y=risk_by_city['risk_pct'],
        marker_color=colors_city,
        text=risk_by_city['risk_pct'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=risk_by_channel['channel'],
        y=risk_by_channel['risk_pct'],
        marker_color=colors_channel,
        text=risk_by_channel['risk_pct'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        showlegend=False
    ), row=1, col=2)
    
    fig.update_layout(
        title=dict(text='Stockout Risk Distribution', font=dict(size=18)),
        height=400
    )
    fig.update_yaxes(title_text='Risk %', row=1, col=1)
    
    return fig


def create_issues_pareto(issues_df: pd.DataFrame) -> go.Figure:
    """Create Pareto chart of data quality issues"""
    if len(issues_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No issues found", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Count by issue type
    issue_counts = issues_df['issue_type'].value_counts().reset_index()
    issue_counts.columns = ['issue_type', 'count']
    issue_counts = issue_counts.sort_values('count', ascending=False)
    
    # Calculate cumulative percentage
    total = issue_counts['count'].sum()
    issue_counts['cumulative_pct'] = (issue_counts['count'].cumsum() / total * 100).round(1)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bar chart
    fig.add_trace(go.Bar(
        x=issue_counts['issue_type'],
        y=issue_counts['count'],
        name='Count',
        marker_color='#4361ee',
        text=issue_counts['count'],
        textposition='outside'
    ), secondary_y=False)
    
    # Cumulative line
    fig.add_trace(go.Scatter(
        x=issue_counts['issue_type'],
        y=issue_counts['cumulative_pct'],
        name='Cumulative %',
        mode='lines+markers',
        line=dict(color='#ef4444', width=2),
        marker=dict(size=8)
    ), secondary_y=True)
    
    fig.update_layout(
        title=dict(text='Issues Pareto Chart', font=dict(size=18)),
        height=400,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    fig.update_yaxes(title_text="Count", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
    
    return fig


def create_simulation_waterfall(summary: Dict) -> go.Figure:
    """Create waterfall chart for simulation financial breakdown"""
    fig = go.Figure(go.Waterfall(
        name="Financial Breakdown",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Revenue", "COGS", "Gross Margin", "Promo Spend", "Net Profit"],
        y=[
            summary['total_simulated_revenue'],
            -summary['total_simulated_cogs'],
            0,
            -summary['total_promo_spend'],
            0
        ],
        textposition="outside",
        text=[
            format_number_short(summary['total_simulated_revenue']),
            format_number_short(-summary['total_simulated_cogs']),
            format_number_short(summary['total_simulated_margin']),
            format_number_short(-summary['total_promo_spend']),
            format_number_short(summary['profit_proxy'])
        ],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#10b981"}},
        decreasing={"marker": {"color": "#ef4444"}},
        totals={"marker": {"color": "#4361ee"}}
    ))
    
    fig.update_layout(
        title=dict(text="Simulation Financial Breakdown", font=dict(size=18)),
        showlegend=False,
        height=400
    )
    
    return fig


# ============== MAIN VIEW RENDERERS ==============
def render_comparison_view():
    """Render raw vs clean data comparison - side by side"""
    st.markdown('<h3 class="section-header">📊 Data Comparison: Raw vs Clean</h3>', unsafe_allow_html=True)
    
    # Create two columns for side-by-side comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: #fef2f2; padding: 1rem; border-radius: 8px; border: 2px solid #fecaca; margin-bottom: 1rem;">
            <h4 style="color: #dc2626; margin: 0;">🔴 Raw Data (With Errors)</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample of raw sales
        raw_sales = st.session_state.raw_data['sales'].head(100)
        display_cols = ['order_id', 'order_time', 'product_id', 'qty', 'selling_price_aed', 'discount_pct']
        available_cols = [c for c in display_cols if c in raw_sales.columns]
        
        st.dataframe(
            raw_sales[available_cols],
            height=350,
            use_container_width=True
        )
        
        # Error summary
        st.markdown("**Detected Issues:**")
        issues = st.session_state.issues_log
        if len(issues) > 0:
            sales_issues = issues[issues['table'] == 'sales']
            top_issues = sales_issues['issue_type'].value_counts().head(5)
            for issue, count in top_issues.items():
                st.markdown(f'<div class="error-item">⚠️ {issue}: {count:,} occurrences</div>', 
                          unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #f0fdf4; padding: 1rem; border-radius: 8px; border: 2px solid #bbf7d0; margin-bottom: 1rem;">
            <h4 style="color: #16a34a; margin: 0;">🟢 Clean Data (Corrected)</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample of clean sales
        clean_sales = st.session_state.clean_data['sales'].head(100)
        available_cols = [c for c in display_cols if c in clean_sales.columns]
        
        st.dataframe(
            clean_sales[available_cols],
            height=350,
            use_container_width=True
        )
        
        # Cleaning summary
        st.markdown("**Cleaning Actions:**")
        if len(issues) > 0:
            action_counts = issues['action_taken'].value_counts()
            for action, count in action_counts.items():
                st.markdown(f'<div class="success-item">✓ {action}: {count:,} records</div>',
                          unsafe_allow_html=True)


def render_manager_view():
    """Render Manager/Operations view"""
    data = get_data()
    filters = st.session_state.get('filters', {})
    issues = st.session_state.issues_log
    
    # Initialize KPI calculator
    kpi_calc = KPICalculator(
        data['sales'], data['products'], data['stores'], 
        data['inventory'], data.get('customers')
    )
    
    # Get all KPIs
    all_kpis = kpi_calc.get_all_kpis(filters, issues)
    
    st.markdown('<h3 class="section-header">📈 Manager Dashboard - Operational KPIs</h3>', unsafe_allow_html=True)
    
    # KPI Cards Row 1 - Operations
    col1, col2, col3, col4 = st.columns(4)
    
    inv_kpis = all_kpis['inventory']
    dq_kpis = all_kpis['data_quality']
    
    with col1:
        status = 'good' if inv_kpis['stockout_rate'] < 5 else 'warning' if inv_kpis['stockout_rate'] < 10 else 'danger'
        render_kpi_card(
            "Stockout Risk %",
            f"{inv_kpis['stockout_rate']:.1f}%",
            status=status
        )
    
    with col2:
        status = 'good' if dq_kpis['return_rate'] < 5 else 'warning' if dq_kpis['return_rate'] < 8 else 'danger'
        render_kpi_card(
            "Return Rate %",
            f"{dq_kpis['return_rate']:.1f}%",
            status=status
        )
    
    with col3:
        status = 'good' if dq_kpis['payment_failure_rate'] < 3 else 'warning' if dq_kpis['payment_failure_rate'] < 5 else 'danger'
        render_kpi_card(
            "Payment Failure %",
            f"{dq_kpis['payment_failure_rate']:.1f}%",
            status=status
        )
    
    with col4:
        render_kpi_card(
            "High-Risk SKUs",
            f"{inv_kpis['low_stock_items']:,}"
        )
    
    # KPI Cards Row 2 - Inventory
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_kpi_card(
            "Total Stock Units",
            f"{inv_kpis['total_stock_units']:,}"
        )
    
    with col2:
        render_kpi_card(
            "Avg Stock/SKU",
            f"{inv_kpis['avg_stock_per_sku']:.0f}"
        )
    
    with col3:
        render_kpi_card(
            "Inventory Turnover",
            f"{inv_kpis['inventory_turnover']:.1f}x"
        )
    
    with col4:
        render_kpi_card(
            "Data Issues",
            f"{dq_kpis['total_data_issues']:,}"
        )
    
    # Charts - FULL WIDTH, ROW BY ROW
    st.markdown("---")
    
    # Chart 1 - Full width
    st.plotly_chart(create_stockout_risk_chart(data), use_container_width=True)
    
    st.markdown("---")
    
    # Chart 2 - Full width
    st.plotly_chart(create_issues_pareto(issues), use_container_width=True)
    
    # Top Risk Items Table
    st.markdown("---")
    st.markdown('<h4 class="section-header">🚨 Top 10 Stockout Risk Items</h4>', unsafe_allow_html=True)
    
    # Calculate risk items
    inventory_df = data['inventory'].copy()
    inventory_df['snapshot_date'] = pd.to_datetime(inventory_df['snapshot_date'])
    latest_inv = inventory_df.sort_values('snapshot_date').groupby(['product_id', 'store_id']).last().reset_index()
    latest_inv = latest_inv.merge(data['stores'][['store_id', 'city', 'channel']], on='store_id', how='left')
    latest_inv = latest_inv.merge(data['products'][['product_id', 'category']], on='product_id', how='left')
    latest_inv['stock_coverage'] = latest_inv['stock_on_hand'] / latest_inv['reorder_point'].replace(0, 1)
    
    risk_items = latest_inv[latest_inv['stock_on_hand'] <= latest_inv['reorder_point']].nsmallest(10, 'stock_coverage')
    
    if len(risk_items) > 0:
        display_cols = ['product_id', 'store_id', 'city', 'channel', 'category', 'stock_on_hand', 'reorder_point']
        available_cols = [c for c in display_cols if c in risk_items.columns]
        st.dataframe(risk_items[available_cols], use_container_width=True, hide_index=True)
    else:
        st.success("No items currently at stockout risk! ✓")
    
    # Department Error Summary
    st.markdown("---")
    st.markdown('<h4 class="section-header">🏢 Issues by Department</h4>', unsafe_allow_html=True)
    
    if len(issues) > 0 and 'department' in issues.columns:
        dept_issues = issues.groupby('department').agg({
            'issue_type': 'count',
            'action_taken': lambda x: (x == 'Corrected').sum()
        }).reset_index()
        dept_issues.columns = ['Department', 'Total Issues', 'Auto-Corrected']
        dept_issues['Manual Review Needed'] = dept_issues['Total Issues'] - dept_issues['Auto-Corrected']
        
        st.dataframe(dept_issues, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Pie chart full width
        fig = px.pie(dept_issues, values='Total Issues', names='Department', 
                    title='Issues Distribution by Department')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_executive_view():
    """Render Executive view"""
    data = get_data()
    filters = st.session_state.get('filters', {})
    issues = st.session_state.issues_log
    
    # Initialize KPI calculator
    kpi_calc = KPICalculator(
        data['sales'], data['products'], data['stores'], 
        data['inventory'], data.get('customers')
    )
    
    # Get all KPIs
    all_kpis = kpi_calc.get_all_kpis(filters, issues)
    biz_kpis = all_kpis['business']
    promo_kpis = all_kpis['promotion']
    channel_kpis = all_kpis['channel']
    
    st.markdown('<h3 class="section-header">📊 Executive Dashboard - Financial KPIs</h3>', unsafe_allow_html=True)
    
    # KPI Cards Row 1 - Finance
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_kpi_card(
            "Net Revenue",
            format_number_short(biz_kpis['net_revenue'])
        )
    
    with col2:
        margin_val = biz_kpis['gross_margin_pct']
        status = 'good' if margin_val >= 25 else 'warning' if margin_val >= 20 else 'danger'
        render_kpi_card(
            "Gross Margin %",
            f"{margin_val:.1f}%",
            status=status
        )
    
    with col3:
        render_kpi_card(
            "Avg Order Value",
            f"AED {biz_kpis['avg_order_value']:,.0f}"
        )
    
    with col4:
        growth = biz_kpis['revenue_growth_pct']
        delta = f"+{growth:.1f}%" if growth > 0 else f"{growth:.1f}%"
        status = 'good' if growth > 0 else 'danger'
        render_kpi_card(
            "Revenue Growth",
            delta,
            status=status
        )
    
    # KPI Cards Row 2 - Promotion
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_kpi_card(
            "Total Orders",
            f"{biz_kpis['total_orders']:,}"
        )
    
    with col2:
        render_kpi_card(
            "Avg Discount %",
            f"{biz_kpis['avg_discount_pct']:.1f}%"
        )
    
    with col3:
        render_kpi_card(
            "Promo ROI",
            f"{promo_kpis['promo_roi']:.1f}x"
        )
    
    with col4:
        util_pct = promo_kpis['budget_utilization_pct']
        status = 'good' if util_pct <= 100 else 'danger'
        render_kpi_card(
            "Budget Utilization",
            f"{util_pct:.0f}%",
            status=status
        )
    
    # Charts - FULL WIDTH, ROW BY ROW
    st.markdown("---")
    
    # Chart 1 - Revenue Trend (Full width)
    st.plotly_chart(create_revenue_trend(data, filters), use_container_width=True)
    
    st.markdown("---")
    
    # Chart 2 - Category Margin (Full width)
    st.plotly_chart(create_category_margin_chart(data), use_container_width=True)
    
    st.markdown("---")
    
    # Chart 3 - Revenue by City (Full width)
    sales_enriched = data['sales'].merge(
        data['stores'][['store_id', 'city']], on='store_id', how='left'
    )
    paid_sales = sales_enriched[sales_enriched['payment_status'] == 'Paid'] if 'payment_status' in sales_enriched.columns else sales_enriched
    city_revenue = paid_sales.groupby('city')['selling_price_aed'].sum().reset_index()
    
    fig = px.bar(city_revenue, x='city', y='selling_price_aed', 
                title='Revenue by City',
                labels={'selling_price_aed': 'Revenue (AED)', 'city': 'City'},
                color='city',
                color_discrete_sequence=['#4361ee', '#f72585', '#4cc9f0'])
    fig.update_layout(showlegend=False, height=400)
    fig.update_traces(texttemplate='AED %{y:,.0f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Chart 4 - Revenue by Channel (Full width)
    sales_enriched = data['sales'].merge(
        data['stores'][['store_id', 'channel']], on='store_id', how='left'
    )
    paid_sales = sales_enriched[sales_enriched['payment_status'] == 'Paid'] if 'payment_status' in sales_enriched.columns else sales_enriched
    channel_revenue = paid_sales.groupby('channel')['selling_price_aed'].sum().reset_index()
    
    fig = px.pie(channel_revenue, values='selling_price_aed', names='channel',
                title='Revenue Share by Channel',
                color_discrete_sequence=['#4361ee', '#f72585', '#4cc9f0'])
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    # Advanced Visualizations - Each in its own row
    st.markdown("---")
    st.markdown('<h4 class="section-header">📊 Strategic Analysis</h4>', unsafe_allow_html=True)
    
    # BCG Matrix (Full width)
    st.plotly_chart(create_bcg_matrix(data), use_container_width=True)
    st.caption("BCG Matrix shows channel positioning based on market share and growth rate. Bubble size represents revenue contribution.")
    
    st.markdown("---")
    
    # Sunburst Chart (Full width)
    st.plotly_chart(create_sunburst_chart(data), use_container_width=True)
    st.caption("Sunburst chart shows hierarchical revenue breakdown from City → Channel → Category.")
    
    st.markdown("---")
    
    # Heatmap (Full width)
    st.plotly_chart(create_heatmap(data), use_container_width=True)
    st.caption("Heatmap shows revenue intensity across categories and channels.")


def render_simulation_view():
    """Render promotional campaign simulation"""
    st.markdown('<h3 class="section-header">🎯 Promotional Campaign Simulator</h3>', unsafe_allow_html=True)
    
    data = get_data()
    
    # Simulation controls
    st.markdown("#### Configure Campaign Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        discount_pct = st.slider("Discount %", 5, 50, 20, key="sim_discount")
    
    with col2:
        promo_budget = st.number_input("Promo Budget (AED)", 50000, 1000000, 200000, 
                                       step=10000, key="sim_budget")
    
    with col3:
        margin_floor = st.slider("Margin Floor %", 5, 30, 15, key="sim_margin_floor")
    
    with col4:
        sim_days = st.selectbox("Simulation Period", [7, 14, 21, 30], index=1, key="sim_days")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cities = ['All'] + list(data['stores']['city'].unique())
        sim_city = st.selectbox("City", cities, key="sim_city")
    
    with col2:
        channels = ['All'] + list(data['stores']['channel'].unique())
        sim_channel = st.selectbox("Channel", channels, key="sim_channel")
    
    with col3:
        categories = ['All'] + list(data['products']['category'].unique())
        sim_category = st.selectbox("Category", categories, key="sim_category")
    
    # Run simulation button
    if st.button("🚀 Run Simulation", type="primary", key="run_sim"):
        with st.spinner("Running simulation..."):
            simulator = PromoSimulator(
                data['sales'], data['products'], 
                data['stores'], data['inventory']
            )
            
            results = simulator.run_simulation(
                discount_pct=discount_pct,
                promo_budget=promo_budget,
                margin_floor_pct=margin_floor,
                simulation_days=sim_days,
                city=sim_city,
                channel=sim_channel,
                category=sim_category
            )
            
            st.session_state.sim_results = results
            st.session_state.simulator = simulator
    
    # Display results
    if 'sim_results' in st.session_state:
        results = st.session_state.sim_results
        summary = results['summary']
        
        st.markdown("---")
        st.markdown("### 📊 Simulation Results")
        
        # KPI Cards for simulation results
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            render_kpi_card(
                "Projected Revenue",
                format_number_short(summary['total_simulated_revenue']),
                f"{sim_days}-day forecast"
            )
        
        with col2:
            margin_pct = summary['total_margin_pct']
            status = 'good' if margin_pct >= margin_floor else 'danger'
            render_kpi_card(
                "Projected Margin",
                f"{margin_pct:.1f}%",
                f"Floor: {margin_floor}%",
                status
            )
        
        with col3:
            render_kpi_card(
                "Promo Spend",
                format_number_short(summary['total_promo_spend']),
                f"Budget: {format_number_short(promo_budget)}"
            )
        
        with col4:
            render_kpi_card(
                "Net Profit",
                format_number_short(summary['profit_proxy']),
                "After promo costs"
            )
        
        with col5:
            risk_pct = summary['stockout_risk_pct']
            status = 'good' if risk_pct < 30 else 'warning' if risk_pct < 50 else 'danger'
            render_kpi_card(
                "Stockout Risk",
                f"{risk_pct:.1f}%",
                f"{summary['items_with_stockout_risk']} items",
                status
            )
        
        # Constraint Violations Alert
        if summary['constraint_violations'] > 0:
            st.warning(f"⚠️ {summary['constraint_violations']} constraint violations detected!")
            
            violations = results['violations']
            if len(violations) > 0:
                with st.expander("View Violation Details"):
                    st.dataframe(violations.head(20), use_container_width=True, hide_index=True)
        
        # Financial Breakdown Chart (Full width)
        st.markdown("---")
        st.plotly_chart(create_simulation_waterfall(summary), use_container_width=True)
        
        # Results by City Table
        st.markdown("---")
        st.markdown("#### 📍 Results by City")
        
        by_city = results.get('by_city', pd.DataFrame())
        if len(by_city) > 0:
            st.dataframe(by_city, use_container_width=True, hide_index=True)
        
        # AI Recommendation
        st.markdown("---")
        st.markdown("### 💡 AI Recommendation")
        
        recommendation = st.session_state.simulator.generate_recommendation(results)
        st.markdown(f"""
        <div class="recommendation-box">
            {recommendation}
        </div>
        """, unsafe_allow_html=True)
        
        # Top Stockout Risk Items
        st.markdown("---")
        st.markdown("#### 🚨 Top Items at Stockout Risk")
        
        risk_items = results.get('top_stockout_risk', pd.DataFrame())
        if len(risk_items) > 0:
            st.dataframe(risk_items.head(10), use_container_width=True, hide_index=True)


def render_error_logs():
    """Render detailed error logs view"""
    st.markdown('<h3 class="section-header">📋 Data Quality Error Logs</h3>', unsafe_allow_html=True)
    
    issues = st.session_state.issues_log
    
    if len(issues) > 0:
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tables = ['All'] + list(issues['table'].unique())
            selected_table = st.selectbox("Filter by Table", tables, key="log_filter_table")
        
        with col2:
            types = ['All'] + list(issues['issue_type'].unique())
            selected_type = st.selectbox("Filter by Issue Type", types, key="log_filter_type")
        
        with col3:
            if 'department' in issues.columns:
                depts = ['All'] + list(issues['department'].unique())
                selected_dept = st.selectbox("Filter by Department", depts, key="log_filter_dept")
            else:
                selected_dept = 'All'
        
        # Filter issues
        filtered_issues = issues.copy()
        if selected_table != 'All':
            filtered_issues = filtered_issues[filtered_issues['table'] == selected_table]
        if selected_type != 'All':
            filtered_issues = filtered_issues[filtered_issues['issue_type'] == selected_type]
        if selected_dept != 'All' and 'department' in filtered_issues.columns:
            filtered_issues = filtered_issues[filtered_issues['department'] == selected_dept]
        
        st.markdown(f"**Showing {len(filtered_issues):,} of {len(issues):,} total issues**")
        
        st.dataframe(
            filtered_issues,
            use_container_width=True,
            hide_index=True,
            height=500
        )
        
        # Download button
        csv = filtered_issues.to_csv(index=False)
        st.download_button(
            "📥 Download Filtered Log",
            csv,
            "filtered_issues.csv",
            "text/csv"
        )
    else:
        st.info("No issues logged. Data appears clean or hasn't been processed yet.")


# ============== MAIN APPLICATION ==============
def main():
    """Main application entry point"""
    init_session_state()
    
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        load_or_generate_data()
    
    # Render sidebar
    render_sidebar()
    
    # Dashboard Header
    st.markdown("""
    <div class="dashboard-intro">
        <h1>📊 Promo Pulse Dashboard</h1>
        <p>UAE Retail Analytics & Promotion Simulator</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Dashboard", 
        "🎯 Simulation", 
        "🔄 Data Comparison",
        "📋 Error Logs"
    ])
    
    with tab1:
        if st.session_state.current_view == 'Manager':
            render_manager_view()
        else:
            render_executive_view()
    
    with tab2:
        render_simulation_view()
    
    with tab3:
        render_comparison_view()
    
    with tab4:
        render_error_logs()


if __name__ == "__main__":
    main()
