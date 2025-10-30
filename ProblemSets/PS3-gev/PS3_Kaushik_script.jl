
# ============================================================
# PS3 – GEV Models (ECON 6343): Main Script
# Student: Bhaskar Kaushik
# Course: ECON 6343 - Econometrics III
# Professor: Tyler Ransom, University of Oklahoma
# AI note (required by syllabus): "Used Claude and ChatGPT to debug and refine code/tests for ECON 6343 Fall 2025 PS3."
# ============================================================

using Random, LinearAlgebra, Statistics
using DataFrames, CSV, HTTP
using Optim
using Distributions

# Load source functions
include("PS3_Kaushik_source.jl")

println("\n", "="^70)
println("PS3 - GEV MODELS ANALYSIS")
println("Student: Bhaskar Kaushik")
println("Course: ECON 6343 - Econometrics III") 
println("University of Oklahoma")
println("="^70, "\n")

# ============================================================
# DATA LOADING AND PREPARATION
# ============================================================

println("Loading data from GitHub repository...")
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"

try
    global df = CSV.read(HTTP.get(url).body, DataFrame)
    println("✓ Data loaded successfully")
    println("  Dimensions: ", size(df))
    println("  Variables: ", names(df))
catch e
    println("✗ Error loading data: ", e)
    exit(1)
end

# ============================================================
# DESIGN MATRIX CONSTRUCTION  
# ============================================================

println("\nConstructing design matrices...")

# X matrix: intercept + individual characteristics
X = Float64.(hcat(ones(size(df,1)),    # intercept (ASC)
                  df.age,              # age
                  df.white,            # white dummy
                  df.collgrad))        # college graduate dummy

# Z matrix: expected log wages for each occupation
Z = Float64.(hcat(df.elnwage1,         # Professional/Technical
                  df.elnwage2,         # Managers/Administrators  
                  df.elnwage3,         # Sales
                  df.elnwage4,         # Clerical/Unskilled
                  df.elnwage5,         # Craftsmen
                  df.elnwage6,         # Operatives
                  df.elnwage7,         # Transport
                  df.elnwage8))        # Other

# y vector: occupation choices (1-8)
y = Int.(df.occupation)

println("✓ Design matrices created successfully:")
println("  X: ", size(X), " (intercept, age, white, collgrad)")  
println("  Z: ", size(Z), " (expected log wages for 8 occupations)")
println("  y: ", size(y), " (occupation choices 1-8)")

# Data validation
@assert size(X, 1) == size(Z, 1) == length(y) "Dimension mismatch"
@assert all(1 ≤ yi ≤ 8 for yi in y) "Invalid occupation codes"
@assert size(X, 2) == 4 "X should have 4 columns"
@assert size(Z, 2) == 8 "Z should have 8 columns"

println("✓ Data validation passed")

# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

println("\n" * "="^70)
println("DESCRIPTIVE STATISTICS")  
println("="^70)

println("Sample characteristics:")
println("  N = ", length(y), " observations")
println("  Mean age = ", round(mean(df.age), digits=1), " years")
println("  Proportion white = ", round(mean(df.white), digits=3))
println("  Proportion college graduate = ", round(mean(df.collgrad), digits=3))

println("\nOccupation distribution:")
occupation_names = [
    "1. Professional/Technical",
    "2. Managers/Administrators", 
    "3. Sales",
    "4. Clerical/Unskilled",
    "5. Craftsmen", 
    "6. Operatives",
    "7. Transport",
    "8. Other"
]

for j in 1:8
    count_j = sum(y .== j)
    pct_j = count_j / length(y) * 100
    println("  $(occupation_names[j]): $(count_j) ($(round(pct_j, digits=1))%)")
end

println("\nExpected log wage summary:")
println("  Mean across occupations: ", round(mean(Z), digits=3))
println("  Std dev across occupations: ", round(std(Z), digits=3))
println("  Min expected log wage: ", round(minimum(Z), digits=3))
println("  Max expected log wage: ", round(maximum(Z), digits=3))

# ============================================================
# MAIN ANALYSIS
# ============================================================

println("\n" * "="^70)
println("BEGINNING ECONOMETRIC ANALYSIS")
println("="^70)

# Run the complete analysis using source functions
allwrap(X, Z, y; df=df)

# ============================================================  
# COMPLETION
# ============================================================

println("\n" * "="^70)
println("ANALYSIS COMPLETED SUCCESSFULLY")
println("All questions answered:")
println("  ✓ Question 1: Multinomial Logit with Alternative-Specific Wages")
println("  ✓ Question 2: Interpretation of γ Coefficient") 
println("  ✓ Question 3: Nested Logit Estimation and IIA Testing")
println("  ✓ Question 4: Function Integration and Output")
println("  ✓ Question 5: Unit Testing (run PS3_Kaushik_tests.jl)")
println("="^70, "\n")

println("Analysis complete. Results displayed above.")
println("To run unit tests: include(\"PS3_Kaushik_tests.jl\")")
println("To push to GitHub: git add PS3_Kaushik_*.jl && git commit -m \"Complete PS3\" && git push")