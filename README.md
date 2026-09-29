# GHCA-Distance-Dictates-Healthcare-Outcomes-Model as of September 29th 2026 (check dependencies at the bottom of the readme)
Open-source geographic accessibility, infrastructure capacity, and health outcome simulation engine. Focused on cancer but applicable to any specialty of medicine in different countries.

# GHCA: Distance Dictates Healthcare Outcomes Model - from Global HC Analytics LLC, Jeff Conroy - Founder & Managing Principal

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://www.python.org/)
[![Status: v1.0-Release](https://img.shields.io/badge/Status-v1.0--Release-orange.svg)]()

> **Distance is a determinant of survival.**  
> The Global Healthcare Access Model (GHCA) is an open-source geographic computation, capability stratification, and epidemiological simulation engine designed to quantify how physical distance and driving times impact health outcomes—focused on oncology, but applicable to any specialty of medicine across different nations.

---

## 📌 Problem Overview

In specialized medicine and oncology, clinical outcomes are often dictated not just by biology or therapy availability, but by **geography**. 

Patients living outside critical travel-time windows face:
* Delayed diagnoses and late-stage presentation
* Increased appointment abandonment and treatment disruption
* Structural exclusion from advanced therapies and clinical trials
* Higher mortality from acute medical complications (e.g., neutropenic fever, severe toxicity)

GHCA provides a unified, country-agnostic framework to map population clusters, compute road-network drive times, model mortality disparities, and simulate infrastructure interventions at regional or national scale.

---

## ✨ Key Features

- **Geographic Routing Engine (`RoutingEngine`):**
  - High-performance vectorized 2D Haversine distance calculations.
  - OSRM (Open Source Routing Machine) Table API integration for real-world driving times.
- **Access & Analytical Layer (`HealthcareAccessAnalyzer`):**
  - Population-weighted mean travel time and distance metrics.
  - Stratified accessibility modeling by center capability (e.g., Cutting-Edge vs. General Specialty).
  - Multivariate OLS regression linking travel metrics and sociodemographic covariates (e.g., rurality) directly to mortality rates.
- **Intervention Simulation Framework:**
  - Scenario modeling for mobile healthcare units (e.g., first-response diagnostic and treatment modules).
  - Drone logistics and aerial sample routing simulations for remote accessibility.
- **Modular Data Architecture (`CountryDataAdapter`):**
  - Decoupled ingestion layer adaptable to US Census TIGER/ACS, WorldPop, HDX, or national health registries.

---

## 📊 Interpreting Model Outputs

### 1. Accessibility & Coverage Metrics
* **`national_coverage_pct` (% Population Covered):** The percentage of the target population residing within a specified travel window (e.g., $\le 90$ minutes) of an appropriate medical center. Lower numbers highlight geographical deserts and structural access gaps.
* **`weighted_mean_travel_metric` (Population-Weighted Travel Burden):** Reflects the true aggregate travel burden on the population by weighting drive times against population density rather than raw geographical land area.
* **`stratified_coverage` (Capability-Tiered Access):** Breaks down accessibility by facility sophistication (e.g., *General Specialty* vs. *Advanced/Cutting-Edge Centers* offering trial protocols or specialized infrastructure), exposing referral disparities.

### 2. Epidemiological Outcome Regression
The model uses multivariate Ordinary Least Squares (OLS) regression to quantify how travel time impacts mortality:
* **`travel_time` Coefficient (`coef`):** Quantifies the estimated increase in mortality rate per unit increase in travel time (e.g., every additional 30 minutes of travel).
* **$p$-value (`P>|t|`):** Confirms whether the relationship between geography and survival outcomes is statistically significant ($p < 0.05$).
* **R-Squared ($R^2$):** Measures the proportion of variance in health outcomes explained by physical access and sociodemographic covariates (such as rurality index).

### 3. Policy Intervention Scenarios
Compares baseline access against intervention strategies (e.g., mobile care units, regional diagnostic hubs, aerial drone delivery):
* **Net Coverage Delta ($\Delta$):** Quantifies the exact population volume brought into safety travel-time thresholds per deployed unit or logistics asset.

---

## 🚀 Quickstart Example

```python
from ghca_access_model import (
    CountryDataAdapter, 
    RoutingEngine, 
    HealthcareAccessAnalyzer
)
import pandas as pd

# 1. Load Data
adapter = CountryDataAdapter()
pop_df, centers_df = adapter.load_data("PK")

# 2. Initialize Routing & Analytical Engines
routing = RoutingEngine(use_osrm=False)  # Set use_osrm=True if OSRM server is active
analyzer = HealthcareAccessAnalyzer(routing)

# 3. Compute OD Matrix & Coverage Thresholds
od_matrix = routing.compute_od_matrix(pop_df, centers_df)
baseline_eval = analyzer.evaluate_coverage(
    pop_df, centers_df, od_matrix, max_threshold=90.0, threshold_type='drive_time_minutes'
)

print(f"National Coverage (<= 90 mins): {baseline_eval['national_coverage_pct']:.2f}%")
print(f"Weighted Mean Drive Time: {baseline_eval['weighted_mean_travel_metric']:.1f} minutes")

# 4. Simulate Intervention (Adding Mobile Response Unit)
mobile_units = pd.DataFrame([
    {
        "center_id": "MOBILE_01", 
        "lat": 35.9000, 
        "lon": 74.3000, 
        "center_type": "MobileFirstResponse",
        "has_cart": False,
        "has_trials": False
    }
])

intervention_eval = analyzer.run_intervention_scenario(
    pop_df, centers_df, mobile_units_df=mobile_units, max_threshold=90.0
)

print(f"Post-Intervention Coverage: {intervention_eval['national_coverage_pct']:.2f}%")
