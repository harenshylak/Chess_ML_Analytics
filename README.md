# Chess_ML_Analytics

# Chess Game Analytics with Machine Learning

This project explores patterns in online chess games using machine learning and statistical analysis.  
Using a Lichess dataset of ~76k games, we investigate:

- How player rating and rating difference influence game outcomes  
- How opening choices differ by rating level  
- Whether more popular openings actually score better  
- How similar or different the result profiles of openings and their sub-variations are  

All analysis is done in a single Jupyter notebook.

---

## Data

The main dataset used is:

- `games_metadata_profile.csv`  
  - One row per game  
  - Example columns:
    - `GameID`, `Event`, `Site`, `Date`, `TimeControl`
    - `White`, `Black`, `WhiteElo`, `BlackElo`
    - `Moves`, `TotalMoves`, `ECO`, `Opening`
    - `Result` (e.g. `1-0`, `0-1`, `1/2-1/2`)

Additional engineered columns (created inside the notebook):

- `Result_class` – numeric encoding of result (0 = Black win, 1 = Draw, 2 = White win)  
- `EloDiff`, `RelativeEloDiff` – rating-based features  
- `BaseOpening` – high-level opening family extracted from `Opening`  
- Various bins / encodings for modelling, e.g. `RelativeEloDiffBin`, `BaseOpening_code`, etc.

---

## Methods & Analyses

The notebook includes:

1. **Data Cleaning & EDA**
   - Rating distribution and basic summary stats
   - Opening popularity overall and by rating group (high / mid / low)

2. **Predictive Modelling**
   - Target: game result (`Result_class`)
   - Models:
     - Decision Tree Classifier  
     - Random Forest Classifier  
   - Feature sets:
     - Baseline: only rating difference (`RelativeEloDiff` / `RelativeEloDiffBin`)
     - Rich: opening family, rating difference, time control, and both players’ ratings

3. **Regression: Opening Popularity vs Results**
   - For each sufficiently popular base opening:
     - Compute White/Black/Draw percentages
     - Run linear regression:
       - Popularity vs White win rate
       - Popularity vs Black win rate
       - Popularity vs Draw rate

4. **Clustering Openings by Result Profiles**
   - Feature: `[White win %, Draw %, Black win %]` per base opening
   - Use K-Means (k found via elbow + silhouette score)
   - Interpret clusters as “white-favoured”, “balanced”, etc.

5. **Variation-Level Analysis**
   - Work at the `Opening` (full variation) level
   - Attach `BaseOpening` and compute:
     - Result percentages per variation
     - Mean pairwise Euclidean distance between variations’ result profiles
   - Identify openings whose sub-variations are:
     - Very similar (stable)
     - Very different (highly diverse/volatile)
   - Cluster base openings by this “variation distance” metric.
