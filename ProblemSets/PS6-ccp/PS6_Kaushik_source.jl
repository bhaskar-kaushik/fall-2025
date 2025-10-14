# Read the file to compute transition probabilities
include("create_grids.jl")


#========================================
QUESTION 1: Data Loading and Reshaping
========================================#

"""
    load_and_reshape_data(url::String)

Load the bus data from PS5 and reshape to long panel format.

# Arguments
- `url::String`: URL to the CSV file (second CSV from PS5)

# Returns
- `df_long::DataFrame`: Long panel format with columns for bus ID, time, and decision variables
- `Xstate::Matrix`: Mileage states for each bus and time period
- `Zstate::Vector`: Route usage states for each bus  
- `Branded::Vector`: Branded indicator for each bus
"""
function load_and_reshape_data(url::String)
    # Load data
    df = CSV.read(HTTP.get(url).body, DataFrame)
    
    # Create bus id variable
    df = @transform(df, :bus_id = 1:size(df,1))
    
    #---------------------------------------------------
    # Reshape from wide to long (done twice because 
    # stack() requires doing one variable at a time)
    #---------------------------------------------------
    
    # First reshape the decision variable (Y1-Y20)
    dfy = @select(df, :bus_id, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10,
                      :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20,
                      :RouteUsage, :Branded)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id, :RouteUsage, :Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long, :time = kron(collect(1:20), ones(size(df,1))))
    select!(dfy_long, Not(:variable))
    
    # Next reshape the odometer variable (Odo1-Odo20)
    dfx = @select(df, :bus_id, :Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, 
                      :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, :Odo15,
                      :Odo16, :Odo17, :Odo18, :Odo19, :Odo20)
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, :time = kron(collect(1:20), ones(size(df,1))))
    select!(dfx_long, Not(:variable))
    
    # Reshape the mileage state variable (Xst1-Xst20)
    dfxst = @select(df, :bus_id, :Xst1, :Xst2, :Xst3, :Xst4, :Xst5, :Xst6, :Xst7,
                        :Xst8, :Xst9, :Xst10, :Xst11, :Xst12, :Xst13, :Xst14, :Xst15,
                        :Xst16, :Xst17, :Xst18, :Xst19, :Xst20)
    dfxst_long = DataFrames.stack(dfxst, Not([:bus_id]))
    rename!(dfxst_long, :value => :Xstate)
    dfxst_long = @transform(dfxst_long, :time = kron(collect(1:20), ones(size(df,1))))
    select!(dfxst_long, Not(:variable))
    
    # Add the route usage state variable (Zst - constant per bus)
    dfzst = @select(df, :bus_id, :Zst)
    
    # Join all reshaped dataframes back together
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time])
    df_long = leftjoin(df_long, dfxst_long, on = [:bus_id, :time])
    df_long = leftjoin(df_long, dfzst, on = [:bus_id])
    sort!(df_long, [:bus_id, :time])
    
    # Get Xstate, Zstate, and Branded as arrays for later use
    Xstate = Matrix(df[:, [Symbol("Xst$i") for i in 1:20]])
    Zstate = Vector(df[:, :Zst])
    Branded = Vector(df[:, :Branded])
    
    return df_long, Xstate, Zstate, Branded
end


#========================================
QUESTION 2: Flexible Logit Estimation
========================================#

"""
    estimate_flexible_logit(df::DataFrame)

Estimate a flexible logit model with fully interacted terms up to 7th order.

# Arguments
- `df::DataFrame`: Long panel data frame

# Returns
- Fitted GLM model object
"""
function estimate_flexible_logit(df::DataFrame)
    flex_logit = glm(@formula(Y ~ Odometer * Odometer * RouteUsage * RouteUsage * Branded * time * time), 
                    df, Binomial(), LogitLink())
    
    return flex_logit
end

#========================================
QUESTION 3: CCP-Based Estimation
========================================#

"""
    construct_state_space(xbin::Int, zbin::Int, xval::Vector, zval::Vector)

Construct the state space grid for all possible states.

# Arguments
- `xbin::Int`: Number of mileage bins
- `zbin::Int`: Number of route usage bins  
- `xval::Vector`: Grid points for mileage
- `zval::Vector`: Grid points for route usage

# Returns
- `state_df::DataFrame`: Data frame with all possible state combinations
"""
function construct_state_space(xbin::Int, zbin::Int, xval::Vector, zval::Vector, xtran::Matrix)
    state_df = DataFrame(
        Odometer = kron(ones(zbin), xval),
        RouteUsage = kron(zval, ones(xbin)),
        Branded = zeros(size(xtran, 1)),
        time = zeros(size(xtran, 1))
    )
    
    return state_df
end

"""
    compute_future_values(state_df::DataFrame, 
                          flex_logit::GeneralizedLinearModel,
                          xtran::Matrix, 
                          xbin::Int, 
                          T::Int, 
                          β::Float64)

Compute future value terms using CCPs from the flexible logit.

# Arguments
- `state_df::DataFrame`: State space data frame
- `flex_logit`: Fitted flexible logit model
- `xtran::Matrix`: State transition matrix
- `xbin::Int`: Number of mileage bins
- `T::Int`: Number of time periods
- `β::Float64`: Discount factor (0.9)

# Returns
- `FV::Array{Float64,3}`: Future value array (states × brand × time)
"""
function compute_future_values(state_df::DataFrame, 
                                flex_logit,
                                xtran::Matrix, 
                                xbin::Int, 
                                zbin::Int, 
                                T::Int, 
                                β::Float64)
    
    # Initialize the future value array
    FV = zeros(xbin*zbin, 2, T+1)
    
    # Nested loops over time and brand
    for t in 2:T
        for b in 0:1
            # Update state_df - use direct assignment instead of @with
            state_df[!, :time] .= t
            state_df[!, :Branded] .= b
            
            # Compute p0 using predict()
            p0 = 1 .- convert(Array{Float64}, predict(flex_logit, state_df))
            
            # Store -β * log.(p0) in FV
            FV[:, b+1, t] = -β * log.(p0)
        end
    end
    
    return FV
end


"""
    compute_fvt1(FV::Array{Float64,3},
                 xtran::Matrix,
                 Xstate::Matrix,
                 Zstate::Vector,
                 xbin::Int,
                 Branded::Vector)

Map future values from state space to actual data.

# Arguments
- `FV::Array{Float64,3}`: Future value array
- `xtran::Matrix`: State transition matrix
- `Xstate::Matrix`: Mileage state for each observation (N × T)
- `Zstate::Vector`: Route usage state for each bus (N)
- `xbin::Int`: Number of mileage bins
- `Branded::Vector`: Branded indicator for each bus (N)

# Returns
- `FVT1::Vector`: Future value term for each observation in long format
"""
function compute_fvt1(FV::Array{Float64,3},
                      xtran::Matrix,
                      Xstate::Matrix,
                      Zstate::Vector,
                      xbin::Int,
                      Branded::Vector)
    
    # Get dimensions
    N = size(Xstate, 1)  # Number of buses
    T = size(Xstate, 2)  # Number of time periods (20)
    
    # Initialize FVT1
    FVT1 = zeros(N, T)
    
    # Loop over observations and time (over states that were visited)
    for i in 1:N
        row0 = (Zstate[i]-1)*xbin + 1
        for t in 1:T
            row1 = row0 + Xstate[i,t] - 1
            # Compute future value contribution
            FVT1[i,t] = (xtran[row1, :] .- xtran[row0, :])⋅FV[row0:row0+xbin-1, Branded[i]+1, t+1]
        end
    end
    
    # Reshape to long format
    fvt1_long = FVT1'[:]
    
    return fvt1_long
end

"""
    estimate_structural_params(df_long::DataFrame, fvt1::Vector)

Estimate structural parameters θ using GLM with offset.

# Arguments
- `df_long::DataFrame`: Long panel data with decision variable
- `fvt1::Vector`: Future value term to use as offset

# Returns
- Fitted GLM model with structural parameters

"""
function estimate_structural_params(df_long::DataFrame, fvt1::Vector)
    # Add future value to data frame
    df_long = @transform(df_long, :fv = fvt1)
    
    # Estimate structural model
    theta_hat = glm(@formula(Y ~ Odometer + Branded), 
                    df_long, 
                    Binomial(), 
                    LogitLink(), 
                    offset=df_long.fv)
    
    return theta_hat
end


#========================================
MAIN WRAPPER FUNCTION
========================================#

"""
    main()

Main wrapper function to estimate the Rust model using CCPs.
This function calls all the helper functions in sequence.
"""
function main()
    println("="^60)
    println("Rust (1987) Bus Engine Replacement Model - CCP Estimation")
    println("="^60)
    
    # Set parameters
    β = 0.9  # Discount factor
    
    # Step 1: Load and reshape data
    println("\nStep 1: Loading and reshaping data...")
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
    df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
    println("Observations: ", nrow(df_long))
    println("First few rows:")
    println(first(df_long, 5))
    
    # Step 2: Estimate flexible logit
    println("\nStep 2: Estimating flexible logit...")
    flexlogitresults = estimate_flexible_logit(df_long)
    println(flexlogitresults)
    
    # Step 3a: Construct state transition matrices
    println("\nStep 3a: Constructing state transition matrices...")
    # Create state grids
    zval, zbin, xval, xbin, xtran = create_grids()
    
    # Step 3b: Construct state space
    println("\nStep 3b: Constructing state space...")
    statedf = construct_state_space(xbin, zbin, xval, zval, xtran)
    
    # Step 3c: Compute future values
    println("\nStep 3c: Computing future values...")
    FV = compute_future_values(statedf, flexlogitresults, xtran, xbin, zbin, 20, β)
    
    # Step 3d: Map to actual data  
    println("\nStep 3d: Mapping future values to data...")
    efvt1 = compute_fvt1(FV, xtran, Xstate, Zstate, xbin, Branded)
    
    # Verify that efvt1 is non-zero
    println("Future value statistics:")
    println("  Mean: ", mean(efvt1))
    println("  Min: ", minimum(efvt1))
    println("  Max: ", maximum(efvt1))
    println("  Non-zero count: ", sum(efvt1 .!= 0))
    
    # Step 3e: Estimate structural parameters using the computed efvt1
    println("\nStep 3e: Estimating structural parameters...")
    estimates = estimate_structural_params(df_long, efvt1)
    
    # Print results
    println("\n" * "="^60)
    println("RESULTS")
    println("="^60)
    println(estimates)
    
    return nothing
end

