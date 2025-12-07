class DemandProfile:
    # ... (Time Curves remain same) ...
    CURVE_FOOD = [0.01]*7 + [0.05, 0.08, 0.1, 0.05, 0.1, 0.1, 0.05, 0.02, 0.05, 0.1, 0.15, 0.1, 0.03, 0.01]
    CURVE_DAYTIME = [0.0]*8 + [0.1, 0.12, 0.12, 0.12, 0.12, 0.12, 0.1, 0.1, 0.1, 0.0] + [0.0]*5
    CURVE_NIGHT = [0.1, 0.1, 0.05] + [0.0]*15 + [0.05, 0.1, 0.2, 0.2, 0.2]

    PRESETS = {
        "food": {
            "base_rate": 20,
            "order_value_mean": 250,
            "order_value_std": 80,
            "max_orders_per_bldg": 50,
            "time_curve": "food",
            "wealth_sensitivity": 0.8, 
            "price_sensitivity": 0.6,
            "weekend_multiplier": 1.3,
            # NEW: Optimization Constraint
            "min_store_spacing_km": 2.0 
        },
        "grocery": {
            "base_rate": 15,
            "order_value_mean": 600,
            "order_value_std": 200,
            "max_orders_per_bldg": 40,
            "time_curve": "food",
            "wealth_sensitivity": 0.5,
            "price_sensitivity": 0.4,
            "weekend_multiplier": 1.5,
            "min_store_spacing_km": 3.0
        },
        "electronics": {
            "base_rate": 0.5,
            "order_value_mean": 15000,
            "order_value_std": 5000,
            "max_orders_per_bldg": 2,
            "time_curve": "daytime",
            "wealth_sensitivity": 2.5,
            "price_sensitivity": 0.1,
            "weekend_multiplier": 1.2,
            "min_store_spacing_km": 8.0 # Stores must be far apart
        },
        "fashion": {
            "base_rate": 8,
            "order_value_mean": 1200,
            "order_value_std": 400,
            "max_orders_per_bldg": 15,
            "time_curve": "daytime",
            "wealth_sensitivity": 1.2,
            "price_sensitivity": 0.8,
            "weekend_multiplier": 1.4,
            "min_store_spacing_km": 5.0
        }
    }

    @staticmethod
    def get_config(category: str, overrides: dict = None):
        category = category.lower()
        if category not in DemandProfile.PRESETS:
            config = DemandProfile.PRESETS["food"].copy()
        else:
            config = DemandProfile.PRESETS[category].copy()
            
        curve_name = config.get("time_curve", "food")
        if curve_name == "daytime": config["tod_probs"] = DemandProfile.CURVE_DAYTIME
        elif curve_name == "night": config["tod_probs"] = DemandProfile.CURVE_NIGHT
        else: config["tod_probs"] = DemandProfile.CURVE_FOOD

        if overrides: config.update(overrides)
        return config