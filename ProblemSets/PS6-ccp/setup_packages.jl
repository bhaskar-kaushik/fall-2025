# setup_packages.jl
# Run this FIRST to install all required packages

println("Installing required packages for implicit state prices analysis...")
println("This may take a few minutes...")

using Pkg

# List of required packages
packages = [
    "DataFrames",
    "CSV", 
    "HTTP",
    "GLM",
    "Statistics",
    "LinearAlgebra",
    "Plots",
    "StatsPlots",
    "PrettyTables",
    "DataFramesMeta",
    "JLD2"
]

println("\nChecking and installing packages:")
println("="^60)

for pkg in packages
    print("  • $pkg ... ")
    try
        # Check if already installed
        Pkg.status(pkg)
        println("✓ already installed")
    catch
        # Install if not found
        println("installing...")
        Pkg.add(pkg)
        println("    ✓ installed")
    end
end

println("\n" * "="^60)
println("All packages installed successfully!")
println("="^60)

println("\nYou can now run:")
println("  include(\"run_complete_analysis.jl\")")