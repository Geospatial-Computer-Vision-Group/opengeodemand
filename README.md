# 🌍 OpenGeoDemand

**OpenGeoDemand** is a production-grade Geospatial Demand Simulation & Logistics Optimization engine.

It combines **OpenStreetMap (OSM)** building footprints with **Government Economic Census Data** to generate hyper-realistic, spatially accurate demand patterns for any Indian city. It goes beyond simple random generation by applying economic elasticity, temporal logic, and agent-based modeling.

> **Status:** v0.2.0 (Stable) | **License:** MIT

---

## 📦 Installation

OpenGeoDemand comes with all necessary economic datasets bundled. You do not need to download external files.

```bash
pip install opengeodemand
```

*Requirements: Python 3.8+*

---

## 🚀 Quick Start (30 Seconds)

Simulate a Food Delivery network in Surat, optimize 5 dark store locations, and generate an analytics dashboard.

```python
from opengeodemand import GeoDemandModel, StoreOptimizer, DemandAnalyzer

# 1. Initialize & Load Data (Auto-fetches OSM & Bundled Econ Data)
city = "Surat, Gujarat, India"
model = GeoDemandModel(city)
model.load_data()

# 2. Enrich Data (Spatial Join + Sampling for speed)
# We sample 2000 buildings for a quick demo. Remove argument for full city.
model.sample_buildings(2000)
model.enrich_data()

# 3. Simulate Demand (7 Days of Food Orders)
# This generates thousands of orders with timestamps, values, and locations
orders = model.simulate(days=7, category="food")
print(f"Generated {len(orders)} orders.")

# 4. Optimize Supply Chain
# Find 5 optimal store locations, keeping them at least 2km apart
optimizer = StoreOptimizer(orders)
stores = optimizer.optimize_locations(n_stores=5, min_spacing_km=2.0)

# 5. Generate Dashboard (HTML Map + Charts)
analyzer = DemandAnalyzer(orders, output_dir="my_results", city_name=city)
analyzer.generate_report(stores_df=stores)

print("Check 'my_results/maps/dashboard.html'!")
```

---

## ⚙️ How It Works (The Core Engine)

The library operates in **4 Stages**:

### 1. Geospatial Extraction (`load_data`)
*   Fetches the city boundary and building polygons live from **OpenStreetMap**.
*   Loads the bundled **Economic Census & HCES** dataset (District-level wealth/consumption data).

### 2. Data Enrichment (`enrich_data`)
*   Performs a **Spatial Join** to link every building to its specific administrative district.
*   **Household Estimation:** Calculates potential households based on building footprint area and levels.
*   **Economic Proxy:** Assigns a "Wealth Score" to every building based on its district's consumption expenditure.

### 3. Simulation Engine (`simulate`)
Demand is calculated using a weighted ensemble of four stochastic models:
1.  **Poisson Process:** Baseline demand based on household density.
2.  **Gravity Model:** Demand gravitates towards economic centroids.
3.  **Agent-Based:** Individual household probability (Bernoulli trials).
4.  **Self-Exciting:** Orders trigger bursts of local activity (clustering).

**The Math:**
$$ \text{Orders} = \text{BaseRate} \times (\text{Households}) \times (\text{Wealth})^{\text{Sensitivity}} \times (\text{TimeFactor}) $$

### 4. Optimization (`optimize_locations`)
Uses **Revenue-Weighted K-Means Clustering** to find centroids of high demand. It then iteratively merges clusters that violate the `min_store_spacing_km` constraint to ensure realistic coverage.

---

## 🎛️ Customization & Scenarios

You don't need to know the math. You can use **Smart Profiles** or override parameters manually.

### 1. Industry Profiles
The library comes with pre-tuned behaviors:

| Category | Behavior Description |
| :--- | :--- |
| `"food"` | High volume, low value. Peaks at Lunch/Dinner. Moderate weekend boost (`1.3x`). |
| `"grocery"` | Moderate volume. High weekend spike (`1.5x`). Spatially clustered. |
| `"electronics"` | Very low volume, high value. High **Wealth Sensitivity** (only rich areas buy). |
| `"fashion"` | High price sensitivity. Daytime peaks. |

### 2. Custom Parameters (`custom_params`)
You can pass a dictionary to `simulate()` to override any logic:

```python
# Example: Simulating a "Luxury Furniture Sale"
orders = model.simulate(
    days=7,
    category="custom",
    custom_params={
        "base_rate": 0.5,            # Very rare orders
        "order_value_mean": 25000,   # High Ticket Size
        "wealth_sensitivity": 2.5,   # Power Law: Rich districts buy 5x-10x more
        "weekend_multiplier": 2.0,   # Sale is on weekend
        "time_curve": "daytime"      # People don't buy sofas at 2 AM
    }
)
```

### Parameter Reference

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `base_rate` | `20` | Daily orders per "average" district unit. Controls total volume. |
| `max_orders_per_bldg` | `50` | Hard cap to prevent outliers in skyscrapers. |
| `wealth_sensitivity` | `1.0` | **Elasticity:** `>1.0` means rich areas demand disproportionately more. `0.0` means wealth doesn't affect volume. |
| `price_sensitivity` | `1.0` | **Elasticity:** `>1.0` means rich areas pay higher prices. `0.0` means fixed price everywhere. |
| `weekend_multiplier` | `1.0` | Multiplier for Saturday/Sunday volume (e.g., `1.5` = +50%). |
| `min_store_spacing_km`| `2.0` | **Optimization:** Minimum distance allowed between two stores. |
| `time_curve` | `"food"` | `"food"` (1pm/8pm peaks), `"daytime"` (9am-6pm), or `"night"` (8pm-2am). |

---

## 📂 Output Files

The library automatically organizes outputs into a structured directory (e.g., `my_results/`):

### 1. `maps/dashboard.html` 🌟
An interactive LeafletJS map containing:
*   **Heatmap:** Visualizes demand density.
*   **Store Markers:** Optimized locations with projected revenue popup.
*   **Summary Card:** Total revenue, order count, and duration.

### 2. `plots/` (Static Analytics)
*   **`temporal_analysis.png`**: Daily order trend + Hourly distribution bar chart.
*   **`economic_analysis.png`**: Order value histogram + Cumulative revenue curve.
*   **`store_performance.png`**: Revenue comparison bar chart for optimized stores.
*   **`simulation_diagnostics.png`**: Demand concentration analysis (Pareto).

### 3. `orders/` (Raw Data)
*   **`orders.csv`**: The raw transactional log.
    *   Columns: `building_idx`, `timestamp`, `lat`, `lon`, `order_value`, `method`.

### 4. `optimization/`
*   **`optimal_stores.csv`**: Lat/Lon and projected metrics for the deployed stores.

### 5. `data/` (Infrastructure)
*   **`buildings_enriched.geojson`**: The OSM polygons with calculated household and economic attributes. Useful for QGIS visualization.
*   **`districts.geojson`**: The intersected economic districts.

---

## 💡 Use Cases

1.  **Q-Commerce / Dark Stores:** Determine where to open the next 5 warehouses to guarantee 10-minute delivery in high-demand zones.
2.  **Retail Expansion:** Compare "Electronics" vs "Fashion" demand to decide which store format fits a specific neighborhood.
3.  **Logistics Stress Testing:** Generate 100,000 orders to test if your routing algorithm can handle the load.
4.  **Academic Research:** Generate realistic synthetic datasets for Smart City projects where real private data is unavailable.

---


**Data Sources:**
*   Map Data © OpenStreetMap contributors.
*   Economic Data: Aggregated from Public Indian Economic Census & HCES reports.