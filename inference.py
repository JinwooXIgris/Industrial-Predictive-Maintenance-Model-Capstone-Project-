"""
inference.py
Production-grade real-time inference pipeline for predictive maintenance.
Includes input validation, physical range warnings, scaling/encoding, and CLI predictions.
"""

import os
import sys
import pickle
import argparse
from typing import Dict, Any, Tuple, Union
import numpy as np
import pandas as pd


# Import project configurations
try:
    import config
    from feature_engineering import engineer_features
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config
    from feature_engineering import engineer_features


# Physical boundary checks for robust data validation
VALIDATION_RANGES = {
    "air_temp": (240.0, 330.0),       # Kelvin
    "proc_temp": (240.0, 340.0),      # Kelvin
    "rot_speed": (500.0, 4000.0),     # RPM
    "torque": (0.0, 120.0),           # Nm
    "tool_wear": (0.0, 300.0),        # minutes
    "machine_type": ["L", "M", "H"]   # Categorical levels
}


class PredictiveMaintenanceInference:
    """
    Inference class to load model artifacts and perform failure risk predictions
    on real-time telemetry inputs.
    """
    def __init__(self, model_name: str = "random_forest"):
        self.model_name = model_name
        self.scaler = None
        self.encoder = None
        self.feature_names = None
        self.model = None
        self.threshold = 0.5  # Default threshold, can be overridden
        
        self.load_artifacts()

    def load_artifacts(self) -> None:
        """
        Loads fitted preprocessing pipelining and classifier models.
        Uses the model's own feature_names_in_ for exact alignment.
        """
        scaler_path = os.path.join(config.MODELS_DIR, "scaler.pkl")
        encoder_path = os.path.join(config.MODELS_DIR, "encoder.pkl")
        meta_path = os.path.join(config.MODELS_DIR, "pipeline_meta.pkl")
        model_path = os.path.join(config.MODELS_DIR, f"{self.model_name}_final_model.pkl")

        # Verify artifact existence
        for path in [scaler_path, encoder_path, meta_path, model_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required artifact not found: '{path}'. "
                    f"Please run model training and pipeline building first."
                )

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(encoder_path, "rb") as f:
            self.encoder = pickle.load(f)
        with open(meta_path, "rb") as f:
            _ = pickle.load(f)  # not needed for feature names anymore

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # ✅ CRITICAL FIX: use the model's exact feature list
        self.feature_names = self.model.feature_names_in_.tolist()

        # Load optimal threshold (if available)
        optimal_threshold_path = os.path.join(config.RESULTS_DIR, f"{self.model_name}_best_threshold.txt")
        if os.path.exists(optimal_threshold_path):
            try:
                with open(optimal_threshold_path, "r") as f:
                    self.threshold = float(f.read().strip())
                print(f"[INFO] Loaded optimized threshold {self.threshold:.3f} for {self.model_name}")
            except Exception:
                pass

    def validate_inputs(self, raw_input: Dict[str, Any]) -> list:
        """
        Performs range and type checking. Returns a list of diagnostic warnings.
        """
        warnings = []
        
        # Type check and raise errors for missing or physical impossibilities
        for key, limits in VALIDATION_RANGES.items():
            if key not in raw_input:
                raise KeyError(f"Missing required parameter: {key}")
                
            val = raw_input[key]
            
            if key == "machine_type":
                if val not in limits:
                    raise ValueError(f"Invalid machine_type '{val}'. Must be one of {limits}.")
            else:
                try:
                    numeric_val = float(val)
                except ValueError:
                    raise TypeError(f"Parameter {key} must be numeric. Received: {val}")
                
                # Check for absolute safety boundaries (negative mechanical properties)
                if key in ["rot_speed", "torque", "tool_wear"] and numeric_val < 0:
                    raise ValueError(f"Parameter {key} cannot be negative. Received: {numeric_val}")
                
                # Append warnings if value is outside typical bounds observed in training
                lower_limit, upper_limit = limits
                if numeric_val < lower_limit or numeric_val > upper_limit:
                    warnings.append(
                        f"Telemetry out-of-bounds warning: {key} value ({numeric_val}) "
                        f"is outside typical training envelope [{lower_limit}, {upper_limit}]."
                    )
                    
        # Check thermodynamic logic
        if float(raw_input["proc_temp"]) < float(raw_input["air_temp"]):
            warnings.append("Physical anomaly: Process temperature is lower than Air temperature.")
            
        return warnings

    def predict(
        self,
        raw_input: Dict[str, Any],
        custom_threshold: float = None
    ) -> Dict[str, Any]:
        """
        Performs preprocessing, feature engineering, and inference.
        
        Parameters:
        -----------
        raw_input : Dict[str, Any]
            Telemetry readings dictionary. Example:
            {
                "air_temp": 298.1,
                "proc_temp": 308.6,
                "rot_speed": 1551.0,
                "torque": 42.8,
                "tool_wear": 0.0,
                "machine_type": "M"
            }
        custom_threshold : float, optional
            Overrides optimal threshold.
            
        Returns:
        --------
        Dict[str, Any]
            Prediction outputs and risk metrics.
        """
        # 1. Run Data Validation
        warnings = self.validate_inputs(raw_input)
        
        # 2. Structure as Single-Row DataFrame for transformations
        input_df = pd.DataFrame([raw_input])
        
        # 3. Apply Feature Engineering
        engineered_df = engineer_features(input_df)
        
        # 4. Scale Numerics and Encode Categoricals using saved pipelines
        numeric_cols = config.CLEAN_NUMERIC_FEATURES + config.ENGINEERED_FEATURES
        scaled_numeric = self.scaler.transform(engineered_df[numeric_cols])
        
        cat_cols = [config.CLEAN_CATEGORICAL_FEATURE]
        encoded_cat = self.encoder.transform(engineered_df[cat_cols])
        
        features_combined = np.hstack([scaled_numeric, encoded_cat])
        X_infer = pd.DataFrame(features_combined, columns=self.model.feature_names_in_)
        
        # 5. Model Prediction
        # Obtain prediction probability (class 1)
        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(X_infer)[0, 1])
        else:
            # Fallback for models without predict_proba (like some linear/margin models)
            decision = self.model.decision_function(X_infer)[0]
            probability = float(1 / (1 + np.exp(-decision))) # Sigmoid squash
            
        # Determine classification threshold
        thresh = custom_threshold if custom_threshold is not None else self.threshold
        prediction = int(probability >= thresh)
        
        # Calculate qualitative risk level
        if probability < 0.15:
            risk_level = "Low"
        elif probability < 0.50:
            risk_level = "Elevated"
        elif probability < 0.80:
            risk_level = "High"
        else:
            risk_level = "Critical"
            
        return {
            "model_used": self.model_name,
            "probability": probability,
            "confidence_pct": probability * 100 if prediction == 1 else (1 - probability) * 100,
            "prediction_class": prediction,
            "prediction_label": "Machine Failure" if prediction == 1 else "Normal Operation",
            "decision_threshold": thresh,
            "risk_level": risk_level,
            "warnings": warnings
        }


if __name__ == "__main__":
    # Command Line Interface execution
    parser = argparse.ArgumentParser(description="Predictive Maintenance Real-Time Inference CLI Tool")
    parser.add_argument("--air_temp", type=float, default=298.1, help="Air Temperature in Kelvin")
    parser.add_argument("--proc_temp", type=float, default=308.6, help="Process Temperature in Kelvin")
    parser.add_argument("--rot_speed", type=float, default=1551.0, help="Rotational Speed in rpm")
    parser.add_argument("--torque", type=float, default=42.8, help="Torque in Nm")
    parser.add_argument("--tool_wear", type=float, default=0.0, help="Tool Wear accumulation in minutes")
    parser.add_argument("--machine_type", type=str, default="M", choices=["L", "M", "H"], help="Machine variant class")
    parser.add_argument("--model", type=str, default="random_forest", help="Trained model variant to run")
    parser.add_argument("--threshold", type=float, default=None, help="Force decision threshold")
    
    args = parser.parse_args()
    
    input_telemetry = {
        "air_temp": args.air_temp,
        "proc_temp": args.proc_temp,
        "rot_speed": args.rot_speed,
        "torque": args.torque,
        "tool_wear": args.tool_wear,
        "machine_type": args.machine_type,
        "power_proxy": args.power_proxy,
        "temp_diff": args.temp_diff,
        "wear_per_speed": args.wear_per_speed
    }
    
    try:
        engine = PredictiveMaintenanceInference(model_name=args.model)
        result = engine.predict(input_telemetry, custom_threshold=args.threshold)
        
        print("\n=== INFERENCE PREDICTION RESULTS ===")
        print(f"Model Utilized     : {result['model_used']}")
        print(f"Failure Probability: {result['probability']:.4f}")
        print(f"Risk Assessment    : {result['risk_level']}")
        print(f"Predicted Class    : {result['prediction_class']} ({result['prediction_label']})")
        print(f"Confidence Level   : {result['confidence_pct']:.2f}%")
        print(f"Threshold Set      : {result['decision_threshold']:.3f}")
        
        if result["warnings"]:
            print("\n[WARNINGS DETECTED]:")
            for w in result["warnings"]:
                print(f" - {w}")
                
    except FileNotFoundError as fnf:
        print(f"\n[ERROR] Missing models: {fnf}")
        print("Please train models first by executing capstone_main.ipynb or model_training.py")
    except Exception as ex:
        print(f"\n[ERROR] Prediction failed: {ex}")

