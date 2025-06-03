# Italian Sovereign CDS vs. Bond Spread Analysis

This repository contains Python scripts for analyzing the relationship between Italian sovereign CDS-implied spreads and bond spreads. Key features include:

- Data preprocessing and visualization  
- Descriptive statistics  
- CDS-bond spread basis calculation  
- Stationarity testing (ADF)  
- Granger causality analysis  
- VAR modeling with Impulse Response Functions (IRF) and Forecast Error Variance Decomposition (FEVD)  
- Rolling correlation analysis  
- Cointegration testing and VECM estimation  

## Project Context

This analysis focuses on Italy but is part of a wider thesis covering six Eurozone countries: Austria, Belgium, France, Germany, Greece, Italy, and Spain. These are grouped into stable (Austria, Belgium, France, Germany) and vulnerable economies (Greece, Italy, Spain) to study differences in CDS and bond spread dynamics.

The thesis investigates which market—CDS or bonds—leads in incorporating sovereign risk and how these relationships vary by economic strength.

The dataset uses historical CDS and bond spreads, defining the CDS basis as the difference between the CDS conventional spread and the 5-year sovereign bond spread, relative to the 5-year German Bund yield.

---

Explore the scripts to replicate or extend this analysis.
