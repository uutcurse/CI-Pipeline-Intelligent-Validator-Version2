# Calibration Report

### Calibration Method Selection
Fitted on VALIDATION only to prevent Test set leakage.
* **Uncalibrated Val Log Loss:** 1.0108
* **Sigmoid (Platt) Val Log Loss:** 1.0040
* **Isotonic Val Log Loss:** 0.9830
* **Selected Method:** Isotonic

### Test Evaluation
* **Uncalibrated Test Log Loss:** 0.9949
* **Calibrated Test Log Loss:** 0.9934
* **Test ECE:** 0.0247

### Analysis
* **Did confidence reliability improve?** Yes. 
* **Did high-confidence errors decrease compared to E06?** Selected model has 10 high-confidence (>0.8) errors, whereas E06 had 14 high-confidence errors.
