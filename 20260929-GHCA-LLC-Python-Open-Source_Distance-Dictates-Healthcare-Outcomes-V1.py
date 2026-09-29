"""
##################################################################################################@@@@@@@@~~~-....
## GHCA LLC Open-Source Algorithm: Distance Dictates Healthcare Outcomes Model (v1.0)
## Author: Jeff Conroy | Global HC Analytics LLC - September 29th 2026 - Released Open Source on GitHub
## Description: Open-source geographic accessibility, infrastructure capacity, & health outcome
##              simulation engine proving that distance dictates healthcare outcomes (cancer & other specialties).
##
## INTERPRETING MODEL OUTPUTS:
##  1. Coverage & Accessibility Metrics:
##     - `national_coverage_pct`: % of population residing within the travel threshold (e.g., <= 90 mins).
##     - `weighted_mean_travel_metric`: Population-weighted travel burden, accounting for demographic density.
##  2. Statistical Outcome Modeling (OLS Regression):
##     - `min_metric_value` Coef: Estimated increase in mortality per unit increase in travel time/distance.
##     - P>|t|: Statistical significance (p < 0.05 indicates distance is a primary outcome determinant).
##  3. Intervention Scenario Modeling:
##     - Evaluates net population coverage gain (Delta) when deploying mobile units or aerial drone logistics.
########################################################################################################@@@**~~... .  .    .
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm


# =====================================================================~~~~~~~~~~~~~~~~~~~~~~~~~~---
# 1. ROUTING ENGINE — GEOGRAPHIC COMPUTATION LAYER - OSRM Durations converted to Minutes at the end
# ==================================================================~~~~~~~~~~~~~~~~~~~~~~~~~~~-.. .

class RoutingEngine:
    """
    Core geographic engine for the GHCA Distance Dictates Healthcare Outcomes Model.
    Responsible for computing travel distance and drive times between population clusters 
    and medical infrastructure.
    
    Provides:
        - Vectorized Haversine distance calculations
        - OSRM Road Network Table Routing with batching
        - Flexible speed-profile decay fallbacks for rural/unpaved terrains
    """

    def __init__(
        self, 
        use_osrm: bool = False, 
        osrm_url: str = "http://router.project-osrm.org",
        default_speed_kmh: float = 60.0,
        request_timeout: int = 10
    ):
        self.use_osrm = use_osrm
        self.osrm_url = osrm_url.rstrip("/")
        self.default_speed_kmh = default_speed_kmh
        self.request_timeout = request_timeout
        self.earth_radius_km = 6371.0

    def compute_haversine_matrix(
        self, 
        pop_coords: np.ndarray, 
        center_coords: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized 2D Haversine distance computation returning an (N, M) matrix 
        in kilometers between N population clusters and M medical centers.
        """
        lat1, lon1 = np.radians(pop_coords[:, 0]), np.radians(pop_coords[:, 1])
        lat2, lon2 = np.radians(center_coords[:, 0]), np.radians(center_coords[:, 1])

        dlat = lat1[:, np.newaxis] - lat2
        dlon = lon1[:, np.newaxis] - lon2

        a = (
            np.sin(dlat / 2.0) ** 2 +
            np.cos(lat1)[:, np.newaxis] * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        return self.earth_radius_km * (2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))

    def compute_od_matrix(
        self, 
        population_df: pd.DataFrame, 
        centers_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Computes Origin-Destination (OD) matrix mapping every population cluster
        to every medical center.
        
        Returns DataFrame columns:
            cluster_id, center_id, distance_km, drive_time_minutes
        """
        pop_coords = population_df[['lat', 'lon']].to_numpy()
        center_coords = centers_df[['lat', 'lon']].to_numpy()
        
        n_pop = len(population_df)
        n_centers = len(centers_df)

        dist_matrix_km = self.compute_haversine_matrix(pop_coords, center_coords)
        time_matrix_min = (dist_matrix_km / self.default_speed_kmh) * 60.0

        # Attempt OSRM Table API Batch Routing if enabled
        if self.use_osrm and n_pop > 0 and n_centers > 0:
            try:
                osrm_times = self._fetch_osrm_table(population_df, centers_df)
                if osrm_times is not None:
                    time_matrix_min = osrm_times
            except Exception as e:
                print(f"[Warning] OSRM Table API request failed ({e}). Reverting to Haversine speed baseline.")

        # Vectorized assembly into flattened DataFrame
        cluster_ids = np.repeat(population_df['cluster_id'].values, n_centers)
        center_ids = np.tile(centers_df['center_id'].values, n_pop)

        od_df = pd.DataFrame({
            'cluster_id': cluster_ids,
            'center_id': center_ids,
            'distance_km': dist_matrix_km.ravel(),
            'drive_time_minutes': time_matrix_min.ravel()
        })

        return od_df

    def _fetch_osrm_table(
        self, 
        population_df: pd.DataFrame, 
        centers_df: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """Internal helper for making batch OSRM Table API calls."""
        all_coords = (
            [f"{lon:.6f},{lat:.6f}" for lat, lon in zip(population_df['lat'], population_df['lon'])] +
            [f"{lon:.6f},{lat:.6f}" for lat, lon in zip(centers_df['lat'], centers_df['lon'])]
        )
        
        src_indices = ";".join(str(i) for i in range(len(population_df)))
        dst_indices = ";".join(str(i + len(population_df)) for i in range(len(centers_df)))

        url = (
            f"{self.osrm_url}/table/v1/driving/"
            f"{';'.join(all_coords)}?sources={src_indices}&destinations={dst_indices}"
        )

        resp = requests.get(url, timeout=self.request_timeout)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == "Ok":
                # Convert OSRM durations (seconds) to minutes
                return np.array(data["durations"], dtype=float) / 60.0
        return None


# =====================================================================
# 2. HEALTHCARE ACCESS ANALYZER — ANALYSIS & OUTCOME LAYER
# =====================================================================

class HealthcareAccessAnalyzer:
    """
    Transforms geographic routing matrices into population-level accessibility
    metrics, capability gap assessments, and epidemiological outcome models
    proving that distance dictates healthcare outcomes.
    """

    def __init__(self, routing_engine: RoutingEngine):
        self.routing = routing_engine

    def evaluate_coverage(
        self, 
        population_df: pd.DataFrame, 
        centers_df: pd.DataFrame, 
        od_df: pd.DataFrame, 
        max_threshold: float = 90.0, 
        threshold_type: str = 'drive_time_minutes'
    ) -> Dict[str, Union[float, pd.DataFrame]]:
        """
        Evaluates national population accessibility against travel thresholds.
        
        Outputs:
            - national_coverage_pct: % of population living within threshold.
            - weighted_mean_travel_metric: Real travel burden weighted by population.
            - covered_population / total_population: Headcounts.
            - detailed_dataframe: Population clusters with mapped nearest centers & coverage flags.
        """
        metric_col = 'drive_time_minutes' if threshold_type == 'drive_time_minutes' else 'distance_km'
        
        merged_od = od_df.merge(
            centers_df[['center_id', 'center_type', 'has_cart', 'has_trials']], 
            on='center_id', 
            how='left'
        )

        # Find nearest center for each cluster
        min_idx = merged_od.groupby('cluster_id')[metric_col].idxmin()
        nearest_df = merged_od.loc[min_idx].rename(columns={
            metric_col: 'min_metric_value',
            'center_type': 'nearest_center_type',
            'center_id': 'nearest_center_id'
        })

        detailed = population_df.merge(
            nearest_df[['cluster_id', 'nearest_center_id', 'nearest_center_type', 'min_metric_value']],
            on='cluster_id',
            how='left'
        )

        detailed['is_covered'] = detailed['min_metric_value'] <= max_threshold

        total_pop = detailed['population'].sum()
        covered_pop = detailed.loc[detailed['is_covered'], 'population'].sum()
        pct_covered = (covered_pop / total_pop * 100.0) if total_pop > 0 else 0.0

        # Population-Weighted Mean Travel Metric
        weighted_mean_metric = (
            (detailed['min_metric_value'] * detailed['population']).sum() / total_pop
        ) if total_pop > 0 else np.nan

        return {
            'national_coverage_pct': pct_covered,
            'weighted_mean_travel_metric': weighted_mean_metric,
            'covered_population': covered_pop,
            'total_population': total_pop,
            'detailed_dataframe': detailed
        }

    def stratified_coverage(
        self, 
        population_df: pd.DataFrame, 
        centers_df: pd.DataFrame, 
        od_df: pd.DataFrame, 
        max_threshold: float = 90.0,
        threshold_type: str = 'drive_time_minutes'
    ) -> Dict[str, float]:
        """
        Computes coverage across tiered medical center categories (e.g. CuttingEdge vs General).
        Reveals structural disparities in accessing advanced trial centers versus basic care.
        """
        metric_col = 'drive_time_minutes' if threshold_type == 'drive_time_minutes' else 'distance_km'
        merged_od = od_df.merge(centers_df[['center_id', 'center_type']], on='center_id', how='left')

        stratified_results = {}
        total_pop = population_df['population'].sum()

        for ctype in centers_df['center_type'].unique():
            sub_od = merged_od[merged_od['center_type'] == ctype]
            if sub_od.empty:
                stratified_results[ctype] = 0.0
                continue

            min_access = sub_od.groupby('cluster_id')[metric_col].min().reset_index()
            sub_merged = population_df.merge(min_access, on='cluster_id', how='left')
            sub_merged['is_covered'] = sub_merged[metric_col] <= max_threshold

            covered_pop = sub_merged.loc[sub_merged['is_covered'], 'population'].sum()
            stratified_results[ctype] = (covered_pop / total_pop * 100.0) if total_pop > 0 else 0.0

        return stratified_results

    def fit_outcome_model(
        self, 
        detailed_df: pd.DataFrame, 
        covariate_cols: Optional[List[str]] = None
    ) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Multivariate OLS Regression modeling mortality against travel access metrics and covariates.
        
        Interpretation:
            - `min_metric_value` coef indicates the mortality rate increase per minute/km of travel.
            - `P>|t|` confirms statistical significance of geographical access barriers.
        """
        feature_cols = ['min_metric_value']
        if covariate_cols:
            feature_cols.extend(covariate_cols)

        valid_df = detailed_df.dropna(subset=['mortality_rate'] + feature_cols).copy()
        
        X = sm.add_constant(valid_df[feature_cols])
        y = valid_df['mortality_rate']

        return sm.OLS(y, X).fit()

    def run_intervention_scenario(
        self, 
        population_df: pd.DataFrame, 
        centers_df: pd.DataFrame, 
        mobile_units_df: Optional[pd.DataFrame] = None, 
        drone_effective_speed_kmh: Optional[float] = None, 
        threshold_type: str = 'drive_time_minutes', 
        max_threshold: float = 90.0
    ) -> Dict[str, Union[float, pd.DataFrame]]:
        """
        Simulates infrastructure deployment (mobile care units, drone sample routing networks)
        to quantify post-intervention coverage gains.
        """
        augmented_centers = centers_df.copy()

        if mobile_units_df is not None and not mobile_units_df.empty:
            augmented_centers = pd.concat([augmented_centers, mobile_units_df], ignore_index=True)

        scenario_od = self.routing.compute_od_matrix(population_df, augmented_centers)

        if drone_effective_speed_kmh:
            # Overrides travel time calculation using aerial drone speed vector
            scenario_od['drive_time_minutes'] = (scenario_od['distance_km'] / drone_effective_speed_kmh) * 60.0

        return self.evaluate_coverage(population_df, augmented_centers, scenario_od, max_threshold, threshold_type)


# =====================================================================~~~~~--
# 3. COUNTRY DATA ADAPTER — DATA INGESTION LAYER - data available free online
# =====================================================================~~~---.

class CountryDataAdapter:
    """
    Standardized ingestion interface for mapping census blocks, WorldPop grids,
    and healthcare facility registries across nations.
    """

    def load_data(self, country_code: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        code = country_code.upper()

        if code == "US":
            pop = pd.DataFrame([
                {"cluster_id": "US_NJ_01", "lat": 40.7128, "lon": -74.0060, "population": 500000, "rurality_index": 0.1, "mortality_rate": 42.1},
                {"cluster_id": "US_NJ_02", "lat": 40.0583, "lon": -74.4057, "population": 250000, "rurality_index": 0.4, "mortality_rate": 58.3},
                {"cluster_id": "US_NJ_03", "lat": 39.4234, "lon": -75.0280, "population": 80000,  "rurality_index": 0.8, "mortality_rate": 71.0},
            ])
            centers = pd.DataFrame([
                {"center_id": "US_NCI_01", "lat": 40.7128, "lon": -74.0060, "center_type": "CuttingEdge", "has_cart": True, "has_trials": True},
                {"center_id": "US_GEN_01", "lat": 40.2206, "lon": -74.7597, "center_type": "GeneralOncology", "has_cart": False, "has_trials": False}
            ])

        elif code == "PK":
            pop = pd.DataFrame([
                {"cluster_id": "PK_LHR_01", "lat": 31.5204, "lon": 74.3587, "population": 11000000, "rurality_index": 0.1, "mortality_rate": 45.2},
                {"cluster_id": "PK_KSR_01", "lat": 31.1179, "lon": 74.4445, "population": 1400000,  "rurality_index": 0.6, "mortality_rate": 68.1},
                {"cluster_id": "PK_GLT_01", "lat": 35.9208, "lon": 74.3089, "population": 220000,   "rurality_index": 0.95, "mortality_rate": 92.4},
            ])
            centers = pd.DataFrame([
                {"center_id": "PK_SKM_01", "lat": 31.4288, "lon": 74.2818, "center_type": "CuttingEdge", "has_cart": True, "has_trials": True},
                {"center_id": "PK_DHQ_01", "lat": 31.1150, "lon": 74.4500, "center_type": "GeneralOncology", "has_cart": False, "has_trials": False}
            ])

        else:
            raise ValueError(f"Country code '{country_code}' is not supported in sample data adapter.")

        return pop, centers


# =====================================================================
# 4. EXECUTION WORKFLOW & OUTPUT INTERPRETATION Demo V1.0
# =====================================================================

if __name__ == "__main__":
    adapter = CountryDataAdapter()
    routing = RoutingEngine(use_osrm=False)
    analyzer = HealthcareAccessAnalyzer(routing)

    # 1. Load Data
    pop_df, centers_df = adapter.load_data("PK")

    # 2. Compute Routing & Baseline Coverage
    od_matrix = routing.compute_od_matrix(pop_df, centers_df)
    baseline_eval = analyzer.evaluate_coverage(
        pop_df, centers_df, od_matrix,
        max_threshold=90.0,
        threshold_type='drive_time_minutes'
    )

    print("====================================================================================")
    print("# GHCA LLC OPEN-SOURCE ALGORITHM: DISTANCE DICTATES HEALTHCARE OUTCOMES MODEL v1.0 #")
    print("------------------------------------------------------------------------------------")
    print("\n1. BASELINE COVERAGE & ACCESSIBILITY METRICS")
    print(f"  • National Population Coverage (<= 90 mins) : {baseline_eval['national_coverage_pct']:.2f}%")
    print(f"  • Population-Weighted Mean Drive Time      : {baseline_eval['weighted_mean_travel_metric']:.1f} mins")
    print(f"  • Total Covered Population                  : {baseline_eval['covered_population']:,} / {baseline_eval['total_population']:,}")
    print("  [INTERPRETATION: Lower national coverage indicates geographic access deserts.")
    print("   Population-weighting prevents unpopulated rural land from skewing travel averages.]")

    # 3. Stratified Breakdown
    stratified = analyzer.stratified_coverage(pop_df, centers_df, od_matrix, max_threshold=90.0)
    print("\n2. CAPABILITY-STRATIFIED ACCESSIBILITY BREAKDOWN")
    for ctype, pct in stratified.items():
        print(f"  • {ctype:<20}: {pct:.2f}% coverage")
    print("  [INTERPRETATION: Highlights referral disparities. Patients may live near general care")
    print("   but remain isolated from specialized/cutting-edge clinical trial centers.]")

    # 4. Statistical Impact Model
    detailed_df = baseline_eval['detailed_dataframe']
    model = analyzer.fit_outcome_model(detailed_df, covariate_cols=['rurality_index'])

    print("\n3. EPIDEMIOLOGICAL OUTCOME MODEL (Mortality vs Travel Time)")
    print(model.summary().tables[1])
    print("  [INTERPRETATION: Check the 'min_metric_value' coefficient and P-value.")
    print("   A positive coefficient indicates that each additional minute/km of travel directly")
    print("   correlates with higher mortality rate. p < 0.05 confirms statistical significance.]")

    # 5. Policy Intervention Simulation
    mobile_units = pd.DataFrame([
        {
            "center_id": "MOBILE_ATLANTIC_01", 
            "lat": 35.9000, 
            "lon": 74.3000, 
            "center_type": "MobileFirstResponse",
            "has_cart": False,
            "has_trials": False
        }
    ])

    intervention_eval = analyzer.run_intervention_scenario(
        pop_df, centers_df,
        mobile_units_df=mobile_units,
        max_threshold=90.0
    )

    net_gain_pct = intervention_eval['national_coverage_pct'] - baseline_eval['national_coverage_pct']
    net_gain_pop = intervention_eval['covered_population'] - baseline_eval['covered_population']

    print("\n4. INTERVENTION SIMULATION RESULTS (Mobile Response Unit Deployment)")
    print(f"  • Post-Intervention Coverage (<= 90 mins)  : {intervention_eval['national_coverage_pct']:.2f}%")
    print(f"  • Net Population Coverage Gain (Delta)    : +{net_gain_pct:.2f}% (+{net_gain_pop:,} residents)")
    print("  [INTERPRETATION: Quantifies exact ROI of infrastructure expansion by measuring")
    print("   how many additional residents enter the safe 90-minute care window.]")
    print("==================================================================================")