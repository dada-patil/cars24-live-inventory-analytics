# cars24-live-inventory-analytics

Used Cars Market — Complete Exploratory Data Analysis (EDA)
Project overview
This project analyzes used-car listings to understand asking prices, displayed prices, mileage, model year, fuel type, transmission, discounts, brands, and cities.

Project objectives
Inspect and clean the used-car dataset.
Calculate meaningful summary statistics and business metrics.
Compare prices across companies, fuel types, transmission types, and cities.
Explore how mileage and vehicle age relate to price.
Visualize patterns and summarize the main findings.
Data note: The attached data was extracted from the output of the uploaded cars.ipynb. It contains 14 unique listings, so findings are exploratory and should not be generalized to the whole used-car market. Price and Display Price are represented in lakh units based on the original scraped labels (₹5.91L, ₹5.61 lakh). Price is treated as the original/listed reference price and Display Price as the displayed price; verify this meaning if the source website changes.

1. Import libraries
Theory: Pandas manages tabular data, NumPy supports numerical operations, Matplotlib creates charts, and Seaborn makes statistical visualizations easier. These libraries are the foundation of the analysis.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
sns.set_theme()
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
     
2. Load the dataset
Theory: Loading data is the first step of EDA. We inspect the shape and first rows to confirm that the file has been read correctly. Keep the CSV in the same folder as this notebook.


data_path = Path("Buy_Used_Cars_Analysis_Data.csv")
if not data_path.exists():
    # If you have your own cleaned file, this fallback accepts the original project filename.
    data_path = Path("Buy_Used_Cars.csv")

df = pd.read_csv(data_path)
print("Dataset loaded from:", data_path)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
display(df.head())
     
3. Understand the data structure
Theory: shape, columns, info(), and describe() provide a quick overview of dataset size, column names, data types, and numerical distributions. This helps identify incorrect data types and unexpected values before analysis.


print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nData types and non-null counts:")
df.info()

print("\nNumerical summary:")
display(df.describe(include="number").T)

print("\nCategorical summary:")
display(df.describe(include="object").T)
     
4. Data quality assessment
Theory: Missing values may distort statistics, while duplicate records can inflate counts and averages. We count missing values and duplicates before deciding how to handle them. We do not automatically delete rows unless they are exact duplicates.


quality_report = pd.DataFrame({
    "Missing Count": df.isna().sum(),
    "Missing Percentage": (df.isna().mean() * 100).round(2),
    "Data Type": df.dtypes.astype(str)
})
display(quality_report.sort_values("Missing Count", ascending=False))
print("Exact duplicate rows:", df.duplicated().sum())
     
5. Clean and standardize columns
Theory: Standardized column names and correct data types make analysis more reliable. Numeric columns are converted with to_numeric(errors='coerce'), which turns invalid values into missing values instead of crashing. Text columns are trimmed to remove extra spaces.


# Standardize column labels and remove accidental whitespace
df.columns = [str(c).strip() for c in df.columns]

# Remove accidental index columns if present
df = df.loc[:, ~df.columns.str.match(r"^Unnamed")].copy()
df = df.drop(columns=["index"], errors="ignore")

# The original notebook used lowercase `model` for the model year.
if "model" in df.columns and "Model Year" not in df.columns:
    df = df.rename(columns={"model": "Model Year"})

# Convert expected numeric fields safely
numeric_candidates = ["Running", "Price", "Display Price", "EMI", "Model Year"]
for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce"
        )

# Trim whitespace from text fields
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace("", pd.NA)

# Remove exact duplicate rows
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print("Duplicate rows removed:", before - len(df))
print("Shape after cleaning:", df.shape)
display(df.head())
     
6. Handle missing values and validate data
Theory: Missingness must be examined before filling or dropping values. For this small listing dataset, we avoid inventing values: missing numeric values remain missing for calculations, and charts use available observations. We also flag impossible or suspicious values for review rather than silently changing them.


print("Missing values after cleaning:")
display(df.isna().sum().to_frame("Missing Count"))

# Basic validation checks (these flags do not automatically remove records)
if "Running" in df.columns:
    print("Negative running values:", int((df["Running"] < 0).sum()))
if "Model Year" in df.columns:
    print("Model years outside 1980–current year + 1:",
          int((~df["Model Year"].between(1980, pd.Timestamp.today().year + 1) & df["Model Year"].notna()).sum()))
if "Price" in df.columns:
    print("Zero or negative reference prices:", int((df["Price"] <= 0).sum()))
if "Display Price" in df.columns:
    print("Zero or negative displayed prices:", int((df["Display Price"] <= 0).sum()))
     
7. Feature engineering
Theory: Feature engineering creates useful variables from existing columns. Here, we calculate approximate vehicle age and price discount. Discount is calculated as reference price minus displayed price; discount percentage uses reference price as the denominator. Because the dataset is small, these are descriptive listing metrics, not proof of the final negotiated sale price.


current_year = pd.Timestamp.today().year

if "Model Year" in df.columns:
    df["Vehicle Age"] = current_year - df["Model Year"]

if {"Price", "Display Price"}.issubset(df.columns):
    df["Discount"] = df["Price"] - df["Display Price"]
    df["Discount %"] = np.where(
        df["Price"] > 0,
        (df["Discount"] / df["Price"]) * 100,
        np.nan
    )

# Build a display label for charts/tables where useful
if {"Company", "Vehicle"}.issubset(df.columns):
    df["Car Name"] = (df["Company"].fillna("Unknown").astype(str) + " " +
                      df["Vehicle"].fillna("Unknown").astype(str)).str.strip()

display(df.head())
     
8. Overall EDA calculations
Theory: Mean, median, range, and standard deviation describe the central value and spread of numeric fields. The median is useful when a few expensive cars make the mean unusually high. Prices remain in the dataset's lakh units, while running is in kilometres and EMI is in the source's monthly amount.


numeric_cols = [c for c in [
    "Price", "Display Price", "Discount", "Discount %",
    "Running", "EMI", "Model Year", "Vehicle Age"
] if c in df.columns]

summary = df[numeric_cols].agg(["count", "mean", "median", "std", "min", "max"]).T
display(summary)

if "Display Price" in df.columns:
    print(f"Average displayed price: {df['Display Price'].mean():.2f} lakh")
    print(f"Median displayed price: {df['Display Price'].median():.2f} lakh")
if "Running" in df.columns:
    print(f"Average running: {df['Running'].mean():,.0f} km")
if "Discount" in df.columns:
    print(f"Average listed discount: {df['Discount'].mean():.2f} lakh")
if "Discount %" in df.columns:
    print(f"Average discount percentage: {df['Discount %'].mean():.2f}%")
     
9. Company-wise price comparison
Theory: Grouping by company helps compare the average and median displayed prices for available brands. Median price is included because a company may have both budget and premium models. Results should be interpreted cautiously when each company has only one or two listings.


if {"Company", "Display Price"}.issubset(df.columns):
    company_summary = (
        df.groupby("Company", dropna=True)
          .agg(Listings=("Display Price", "count"),
               Average_Price=("Display Price", "mean"),
               Median_Price=("Display Price", "median"))
          .sort_values("Average_Price", ascending=False)
          .round(2)
    )
    display(company_summary)
     
10. Fuel type and transmission analysis
Theory: Category counts show which kinds of cars appear most often in this sample. Grouped average prices help compare listed prices by fuel type or transmission, but they do not control for model year, mileage, or brand.


if "Fuel Type" in df.columns:
    fuel_summary = df.groupby("Fuel Type", dropna=True).agg(
        Listings=("Fuel Type", "size"),
        Average_Price=("Display Price", "mean"),
        Median_Price=("Display Price", "median")
    ).sort_values("Listings", ascending=False).round(2)
    display(fuel_summary)

if "Transmission" in df.columns:
    transmission_summary = df.groupby("Transmission", dropna=True).agg(
        Listings=("Transmission", "size"),
        Average_Price=("Display Price", "mean"),
        Median_Price=("Display Price", "median")
    ).sort_values("Listings", ascending=False).round(2)
    display(transmission_summary)
     
11. Best-discount and high-mileage listings
Theory: Sorting by discount percentage identifies listings with the largest relative difference between the reference and displayed prices. Sorting by running identifies the most-used cars in the sample. A high discount alone does not mean a car is the best deal; condition, ownership, accident history, and service records are not captured here.


if {"Car Name", "Discount", "Discount %"}.issubset(df.columns):
    print("Top listings by discount percentage:")
    display(df.sort_values("Discount %", ascending=False)[
        [c for c in ["Car Name", "Model Year", "Price", "Display Price", "Discount", "Discount %", "Running"]
         if c in df.columns]
    ].head(10).round(2))

if "Running" in df.columns:
    print("Highest-running listings:")
    display(df.sort_values("Running", ascending=False)[
        [c for c in ["Car Name", "Model Year", "Running", "Display Price", "Fuel Type", "City"]
         if c in df.columns]
    ].head(10))
     
12. Price distribution — histogram
Theory: A histogram groups numeric prices into bins and shows how many listings fall within each price range. It helps identify whether this sample is concentrated in lower-priced cars or spread across several price levels.


if "Display Price" in df.columns:
    plt.figure(figsize=(9, 5))
    sns.histplot(data=df, x="Display Price", bins=min(8, max(3, df["Display Price"].dropna().nunique())), kde=True, color="steelblue")
    plt.title("Distribution of Displayed Used-Car Prices")
    plt.xlabel("Displayed Price (lakh)")
    plt.ylabel("Number of Listings")
    plt.tight_layout()
    plt.show()
     
13. Average price by company — horizontal bar chart
Theory: A horizontal bar chart makes it easier to compare brands. The chart displays the average displayed price for each company, ordered from highest to lowest. Use the listing-count table alongside it to avoid over-interpreting averages based on very few cars.


if {"Company", "Display Price"}.issubset(df.columns):
    company_avg = df.groupby("Company")["Display Price"].mean().sort_values()
    plt.figure(figsize=(9, max(4, 0.45 * len(company_avg))))
    company_avg.plot(kind="barh")
    plt.title("Average Displayed Price by Company")
    plt.xlabel("Average Displayed Price (lakh)")
    plt.ylabel("Company")
    plt.tight_layout()
    plt.show()
     
14. Listing counts by fuel type
Theory: A count plot compares the number of observations in each fuel category. This describes the composition of the current dataset, not the overall market share of each fuel type.


if "Fuel Type" in df.columns:
    plt.figure(figsize=(8, 5))
    order = df["Fuel Type"].value_counts().index
    sns.countplot(data=df, x="Fuel Type", order=order, color="steelblue")
    plt.title("Used-Car Listings by Fuel Type")
    plt.xlabel("Fuel Type")
    plt.ylabel("Number of Listings")
    plt.tight_layout()
    plt.show()
     
15. Displayed price by fuel type — box plot
Theory: A box plot compares the median, spread, and possible outliers of prices across categories. The chart can show whether one fuel category tends to have higher-priced listings in this sample, but sample sizes may be small.


if {"Fuel Type", "Display Price"}.issubset(df.columns):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Fuel Type", y="Display Price")
    plt.title("Displayed Price by Fuel Type")
    plt.xlabel("Fuel Type")
    plt.ylabel("Displayed Price (lakh)")
    plt.tight_layout()
    plt.show()
     
16. Running distance vs displayed price — scatter plot
Theory: A scatter plot places one listing at each mileage-price pair. It helps inspect whether higher running is associated with lower prices. The relationship may also depend on model year, brand, and condition, so correlation is not causation.


if {"Running", "Display Price"}.issubset(df.columns):
    plot_df = df.dropna(subset=["Running", "Display Price"])
    plt.figure(figsize=(9, 5))
    if "Fuel Type" in plot_df.columns:
        sns.scatterplot(data=plot_df, x="Running", y="Display Price", hue="Fuel Type", s=90)
    else:
        sns.scatterplot(data=plot_df, x="Running", y="Display Price", s=90)
    plt.title("Running Distance vs Displayed Price")
    plt.xlabel("Running Distance (km)")
    plt.ylabel("Displayed Price (lakh)")
    plt.tight_layout()
    plt.show()
     
17. Vehicle age vs displayed price — scatter plot
Theory: Vehicle age is approximated by current year minus model year. A scatter plot helps explore whether newer cars have higher displayed prices. This is only a rough comparison because vehicle model, trim, mileage, and condition differ.


if {"Vehicle Age", "Display Price"}.issubset(df.columns):
    plot_df = df.dropna(subset=["Vehicle Age", "Display Price"])
    plt.figure(figsize=(9, 5))
    sns.scatterplot(data=plot_df, x="Vehicle Age", y="Display Price", hue="Transmission" if "Transmission" in plot_df.columns else None, s=90)
    plt.title("Vehicle Age vs Displayed Price")
    plt.xlabel("Approximate Vehicle Age (years)")
    plt.ylabel("Displayed Price (lakh)")
    plt.tight_layout()
    plt.show()
     
18. Discount analysis
Theory: Discount amount and discount percentage answer different questions. Amount shows the absolute difference in lakh units; percentage compares the discount relative to the reference price. This chart helps compare how large the displayed reductions are.


if "Discount %" in df.columns:
    discount_df = df.dropna(subset=["Discount %"]).sort_values("Discount %", ascending=False)
    labels = discount_df["Car Name"] if "Car Name" in discount_df.columns else discount_df.index.astype(str)
    plt.figure(figsize=(10, max(4, 0.45 * len(discount_df))))
    plt.barh(labels, discount_df["Discount %"])
    plt.gca().invert_yaxis()
    plt.title("Discount Percentage by Listing")
    plt.xlabel("Discount (%)")
    plt.ylabel("Car")
    plt.tight_layout()
    plt.show()
     
19. EMI vs displayed price
Theory: This scatter plot explores whether higher displayed prices are associated with higher monthly EMI values. EMI also depends on loan duration, interest rate, down payment, and lender terms, which are not included in this dataset.


if {"EMI", "Display Price"}.issubset(df.columns):
    plt.figure(figsize=(9, 5))
    sns.scatterplot(data=df, x="Display Price", y="EMI", hue="Fuel Type" if "Fuel Type" in df.columns else None, s=90)
    plt.title("Displayed Price vs Monthly EMI")
    plt.xlabel("Displayed Price (lakh)")
    plt.ylabel("EMI (source amount per month)")
    plt.tight_layout()
    plt.show()
     
20. Listings by city
Theory: A city count plot identifies where the observed listings are located. It describes only the locations captured in this dataset and should not be interpreted as a ranking of city-wide supply.


if "City" in df.columns:
    city_counts = df["City"].value_counts().head(10).sort_values()
    plt.figure(figsize=(9, max(4, 0.45 * len(city_counts))))
    city_counts.plot(kind="barh")
    plt.title("Number of Listings by City (Top 10)")
    plt.xlabel("Number of Listings")
    plt.ylabel("City")
    plt.tight_layout()
    plt.show()
     
21. Correlation analysis
Theory: Correlation measures the strength and direction of a linear relationship between numeric variables. Values near +1 indicate positive association, values near -1 indicate negative association, and values near 0 indicate little linear association. With only 14 listings, correlations can change substantially when new rows are added.


corr_cols = [c for c in ["Price", "Display Price", "Discount", "Discount %", "Running", "EMI", "Model Year", "Vehicle Age"]
             if c in df.columns]
corr_df = df[corr_cols].corr(numeric_only=True)
display(corr_df.round(2))

plt.figure(figsize=(9, 6))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.show()
     
22. Outlier review using the IQR method
Theory: The interquartile range (IQR) is Q3 minus Q1. Values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR are flagged as potential outliers. Outliers are not automatically errors; premium cars may legitimately have much higher prices.


for col in [c for c in ["Display Price", "Running", "EMI"] if c in df.columns]:
    s = df[col].dropna()
    if len(s) >= 4:
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flagged = df.loc[df[col].lt(lower) | df[col].gt(upper), [c for c in ["Car Name", col, "Company", "City"] if c in df.columns]]
        print(f"{col}: IQR lower bound={lower:.2f}, upper bound={upper:.2f}; flagged rows={len(flagged)}")
        display(flagged)
     
23. Key findings summary
Theory: A conclusion section should summarize results produced by the dataset, rather than making unsupported market-wide claims. The code below generates simple data-driven statements from the available rows.


print("KEY FINDINGS")
print("-" * 60)
print(f"Total unique listings analyzed: {len(df)}")

if "Display Price" in df.columns and df["Display Price"].notna().any():
    cheapest = df.loc[df["Display Price"].idxmin()]
    costliest = df.loc[df["Display Price"].idxmax()]
    print(f"Average displayed price: {df['Display Price'].mean():.2f} lakh")
    print(f"Median displayed price: {df['Display Price'].median():.2f} lakh")
    print(f"Lowest displayed price: {cheapest['Display Price']:.2f} lakh")
    print(f"Highest displayed price: {costliest['Display Price']:.2f} lakh")
    if "Car Name" in df.columns:
        print("Lowest-priced listing:", cheapest["Car Name"])
        print("Highest-priced listing:", costliest["Car Name"])

if "Discount %" in df.columns and df["Discount %"].notna().any():
    best_discount = df.loc[df["Discount %"].idxmax()]
    print(f"Largest discount percentage: {best_discount['Discount %']:.2f}%")
    if "Car Name" in df.columns:
        print("Listing with largest relative discount:", best_discount["Car Name"])

if "Fuel Type" in df.columns and df["Fuel Type"].notna().any():
    print("Most common fuel type:", df["Fuel Type"].mode().iloc[0])
if "Company" in df.columns and df["Company"].notna().any():
    print("Most frequently listed company:", df["Company"].mode().iloc[0])

print("\nInterpretation caution: this sample contains only 14 listings. Treat findings as exploratory.")
     
24. Export the analysis-ready dataset
Theory: Saving the cleaned dataset allows the results to be reused in another notebook, dashboard, or report. The exported file includes engineered columns such as Vehicle Age, Discount, and Discount %.


output_path = Path("Buy_Used_Cars_Cleaned_EDA.csv")
df.to_csv(output_path, index=False)
print("Cleaned dataset saved to:", output_path.resolve())
display(df.head())
     
25. Project conclusion
This EDA project inspects used-car listing data, standardizes data types, creates vehicle-age and discount features, compares listing prices across categories, and explores relationships among price, mileage, EMI, and age. The visualizations help identify patterns in the available sample. Since the dataset contains only 14 listings and does not include vehicle condition, ownership history, service records, or final sale prices, conclusions should be treated as preliminary rather than representative of the entire used-car market.
