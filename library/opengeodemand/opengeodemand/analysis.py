import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster

class DemandAnalyzer:
    def __init__(self, orders_df, output_dir, city_name="Unknown City"):
        self.orders = orders_df.copy()
        self.orders['timestamp'] = pd.to_datetime(self.orders['timestamp'])
        self.out_dir = output_dir
        self.city_name = city_name
        
        # Style settings
        sns.set_theme(style="whitegrid")
        
        # Ensure subfolders exist
        os.makedirs(os.path.join(self.out_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "maps"), exist_ok=True)

    def generate_report(self, stores_df=None):
        print("  Generating advanced analytics report...")
        
        # 1. Static Plots
        self.plot_temporal_dashboard()
        self.plot_economic_dashboard()
        self.plot_simulation_diagnostics()
        
        if stores_df is not None:
            self.plot_store_performance(stores_df)
            
        # 2. Interactive Map
        self.create_interactive_dashboard(stores_df)

    def plot_temporal_dashboard(self):
        """Creates a dashboard for Time analysis."""
        df = self.orders.set_index('timestamp')
        daily = df.resample('D').size()
        hourly = df.groupby(df.index.hour).size()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Daily Trend
        ax1.plot(daily.index, daily.values, marker='o', color='#2E86C1', linewidth=2)
        ax1.set_title("Daily Order Volatility", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Total Orders")
        ax1.set_xlabel("Date")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Hourly Distribution
        sns.barplot(x=hourly.index, y=hourly.values, ax=ax2, palette="viridis")
        ax2.set_title("Hourly Demand Profile (Aggregate)", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Hour of Day")
        ax2.set_ylabel("Avg Volume")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "plots", "01_temporal_analysis.png"), dpi=150)
        plt.close()

    def plot_economic_dashboard(self):
        """Creates a dashboard for Value/Money analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Order Value Histogram
        sns.histplot(self.orders['order_value'], bins=40, kde=True, color='#8E44AD', ax=ax1)
        ax1.set_title("Basket Size Distribution", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Order Value (INR)")
        ax1.axvline(self.orders['order_value'].mean(), color='red', linestyle='--', label='Mean')
        ax1.legend()
        
        # Plot 2: Cumulative Revenue
        df_sorted = self.orders.sort_values('timestamp')
        df_sorted['cumulative_revenue'] = df_sorted['order_value'].cumsum()
        ax2.fill_between(range(len(df_sorted)), df_sorted['cumulative_revenue'], color='#2ECC71', alpha=0.4)
        ax2.plot(range(len(df_sorted)), df_sorted['cumulative_revenue'], color='#27AE60')
        ax2.set_title("Cumulative Revenue Growth", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Total Revenue (INR)")
        ax2.set_xlabel("Order Count")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "plots", "02_economic_analysis.png"), dpi=150)
        plt.close()

    def plot_simulation_diagnostics(self):
        """Analyzes internal simulation metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Method Contribution
        if 'method' in self.orders.columns:
            counts = self.orders['method'].value_counts()
            ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
            ax1.set_title("Order Generation Method", fontsize=14, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, "Method data not found", ha='center')
            
        # Plot 2: Demand Concentration (Top buildings vs Rest)
        bldg_counts = self.orders['building_idx'].value_counts().values
        ax2.plot(range(len(bldg_counts)), bldg_counts, color='#E74C3C')
        ax2.set_title("Demand Concentration (Orders per Building)", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Orders")
        ax2.set_xlabel("Ranked Buildings (1 = Highest Demand)")
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "plots", "03_simulation_diagnostics.png"), dpi=150)
        plt.close()

    def plot_store_performance(self, stores_df):
        """Analyzes the optimized stores."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Bar chart of revenue per store
        sns.barplot(x='store_id', y='projected_revenue', data=stores_df, ax=ax, palette="Blues_d")
        
        ax.set_title("Projected Revenue per Optimized Store Location", fontsize=14, fontweight='bold')
        ax.set_ylabel("Revenue (INR)")
        ax.set_xlabel("Store ID")
        
        # Format Y axis to Millions
        vals = ax.get_yticks()
        ax.set_yticklabels([f'₹{x/1e6:.1f}M' for x in vals])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "plots", "04_store_performance.png"), dpi=150)
        plt.close()

    def create_interactive_dashboard(self, stores_df=None):
        """Creates a rich HTML map with Titles, Legends, and Layers."""
        center_lat = self.orders['lat'].mean()
        center_lon = self.orders['lon'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='cartodbpositron')

        # --- 1. TITLE CARD (Updated) ---
        total_rev = self.orders['order_value'].sum()
        total_ord = len(self.orders)
        
        # Calculate Date Range dynamically
        min_date = self.orders['timestamp'].min()
        max_date = self.orders['timestamp'].max()
        duration = (max_date - min_date).days + 1
        
        title_html = f'''
             <div style="position: fixed; 
             top: 10px; right: 10px; width: 320px; height: 150px; 
             z-index:9999; font-size:14px;
             background-color: white; opacity: 0.95;
             border: 2px solid #555; border-radius: 8px; padding: 12px;
             box-shadow: 3px 3px 10px rgba(0,0,0,0.2);">
             
             <h3 style="margin-top:0; color:#2C3E50; border-bottom:1px solid #ccc; padding-bottom:5px;">
                GeoDemand Analytics
             </h3>
             
             <b>City:</b> {self.city_name}<br>
             <b>Duration:</b> {duration} Days ({min_date.strftime('%b %d')} - {max_date.strftime('%b %d')})<br>
             
             <div style="margin-top:8px;">
                 <b>Total Revenue:</b> <span style="color:green">₹{total_rev:,.0f}</span><br>
                 <b>Total Orders:</b> {total_ord:,}<br>
                 <b>Optimized Stores:</b> {len(stores_df) if stores_df is not None else 0}
             </div>
             </div>
             '''
        m.get_root().html.add_child(folium.Element(title_html))

        # --- 2. LAYERS ---
        
        # Heatmap
        heat_data = self.orders[['lat', 'lon', 'order_value']].values.tolist()
        hm = HeatMap(heat_data, radius=12, blur=18, name="Demand Heatmap", max_zoom=1)
        hm.add_to(m)

        # Stores
        if stores_df is not None:
            store_group = folium.FeatureGroup(name="Optimized Stores")
            for idx, row in stores_df.iterrows():
                rev = row['projected_revenue']
                rev_str = f"₹{rev/1e6:.2f}M" if rev > 1e6 else f"₹{rev/1e3:.0f}K"
                
                popup_html = f"""
                <div style="font-family: sans-serif; min-width: 150px;">
                    <h4 style="margin:0; background-color:#2ECC71; color:white; padding:5px;">
                        Store #{int(row['store_id'])}
                    </h4>
                    <div style="padding:10px;">
                        <b>Total Revenue ({duration} Days):</b><br>
                        <span style="font-size:16px; color:#27AE60;">{rev_str}</span><br><br>
                        <b>Total Volume:</b><br>
                        {int(row['projected_orders'])} orders
                    </div>
                </div>
                """
                
                folium.Marker(
                    location=[row['lat'], row['lon']],
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"Store #{int(row['store_id'])} ({rev_str})",
                    icon=folium.Icon(color='green', icon='star', prefix='fa')
                ).add_to(store_group)
            store_group.add_to(m)

        folium.LayerControl().add_to(m)

        out_path = os.path.join(self.out_dir, "maps", "dashboard.html")
        m.save(out_path)