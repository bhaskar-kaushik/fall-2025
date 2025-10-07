################################################################################
# RUST (1987) BUS ENGINE REPLACEMENT MODEL - DYNAMIC DISCRETE CHOICE
################################################################################
#
# CRITICAL MATHEMATICAL INSIGHT (from lecture notes):
#
# The Bellman equation for dynamic discrete choice:
#   v_j(X_t) = u_j(X_t) + β·E_{X_{t+1}|X_t,j}[EMAX(X_{t+1})]
#
# where EMAX(X) = E_ε[max_k{v_k(X) + ε_k}] = log(Σ_k exp(v_k(X))) + γ
#
# KEY CONVENTION IN THIS CODE:
# - FV (Future Value array) stores β·EMAX, not just EMAX
# - This means FV already has the discount factor β "baked in"
# - Therefore, when we USE FV[t+1] to compute v_j(X_t), we add it WITHOUT β
# - When we STORE FV[t], we multiply by β: FV[t] = β·(log(Σ exp(v_k)) + γ)
#
# Why this matters:
# - In backward recursion: v_j = u_j + xtran' * FV  (no β, it's in FV!)
# - In likelihood: v_diff = flow_diff + ev_diff  (no β, it's in FV!)
#
# This convention ensures we don't accidentally apply β twice.
#
################################################################################

################################################################################
# PART 1: DATA LOADING AND PREPARATION (Pre-implemented)
################################################################################

"""
    load_static_data()

Load and reshape data for static estimation (Questions 1-2).
Returns a long-format DataFrame ready for GLM estimation.
"""
function load_static_data()
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    
    # Create bus id variable
    df = @transform(df, :bus_id = 1:size(df,1))
    
    # Reshape from wide to long
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
    
    # Join reshaped dataframes back together
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time])
    sort!(df_long, [:bus_id, :time])
    
    return df_long
end

"""
    load_dynamic_data()

Load and prepare data for dynamic estimation (Question 3+).
Returns a named tuple with all data structures needed for estimation.
"""
function load_dynamic_data()
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    
    # Extract matrices and vectors
    Y = Matrix(df[:, r"^Y\d+"])       # Decision variables (N × T)
    X = Matrix(df[:, r"^Odo\d+"])     # Odometer readings (N × T)
    Xstate = Matrix(df[:, r"^Xst"])   # Discretized mileage state (N × T)
    Zstate = Vector(df[:, :Zst])      # Discretized route usage state (N × 1)
    B = Vector(df[:, :Branded])       # Brand indicator (N × 1)
    
    N, T = size(Y)
    
    # Create state grids
    zval, zbin, xval, xbin, xtran = create_grids()
    
    # Bundle everything into a named tuple
    return (
        # Data
        Y = Y,
        X = X,
        B = B,
        Xstate = Xstate,
        Zstate = Zstate,
        # Dimensions
        N = N,
        T = T,
        # State space
        xval = xval,
        xbin = xbin,
        zbin = zbin,
        xtran = xtran,
        # Parameters
        β = 0.9
    )
end

################################################################################
# PART 2: STATIC ESTIMATION (Question 2)
################################################################################

function estimate_static_model(df_long)
    # Estimate the logit model
    theta_static = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink())
    
    println("\n" * "="^70)
    println("STATIC MODEL RESULTS")
    println("="^70)
    println(theta_static)
    
    # Extract and interpret coefficients
    θ = coef(theta_static)
    
    println("\n" * "-"^70)
    println("ECONOMIC INTERPRETATION (Static/Myopic Model, β=0)")
    println("-"^70)
    println("This model assumes Harold Zurcher is completely myopic:")
    println("He only considers TODAY'S utility, not future consequences.")
    println()
    println("Parameter Estimates:")
    println("  θ₀ (constant):     ", round(θ[1], digits=4))
    println("     → Baseline utility of continuing (not replacing)")
    println()
    println("  θ₁ (mileage):      ", round(θ[2], digits=4))
    println("     → Higher mileage REDUCES utility of continuing")
    println("     → Each 10,000 mile increase reduces log-odds by ", round(abs(θ[2]), digits=4))
    println("     → Buses with high mileage are MORE likely to be replaced")
    println()
    println("  θ₂ (brand):        ", round(θ[3], digits=4))
    println("     → Branded (high-end) buses have HIGHER utility of continuing")
    println("     → Better brands are LESS likely to need replacement")
    println("     → Quality matters: premium buses last longer")
    println("-"^70)
    
    return theta_static
end

################################################################################
# PART 3: DYNAMIC ESTIMATION - FUTURE VALUE COMPUTATION (Question 3c)
################################################################################

# compute_future_value!(FV, θ, d)
#
# Compute future value function via backward recursion for all states.
#
# KEY INSIGHT FROM LECTURE NOTES:
# - Conditional value: v_j(X_t) = u_j(X_t) + β·E[EMAX(X_{t+1}) | X_t, j]
# - EMAX (if ε ~ EV1): EMAX(v) = log(Σ exp(v_k)) + γ (Euler's constant)
# - FV stores β·EMAX, so when we use FV[t+1], we add it WITHOUT multiplying by β
#   (β is already "baked into" FV from the previous iteration)
#
# Algorithm:
# 1. Terminal condition: FV[T+1] = 0 (no future)
# 2. Work backward from t=T to t=1
# 3. For each state (z,x,b,t):
#    - v_1 = flow_utility + xtran' * FV[t+1]  [no β! it's in FV]
#    - v_0 = 0 + xtran' * FV[t+1]  [mileage resets to 0]
#    - FV[t] = β·(log(exp(v_0) + exp(v_1)) + γ)  [store discounted EMAX]

@views @inbounds function compute_future_value!(FV, θ, d)
    # FV is already initialized to zeros in the calling function
    # FV[T+1] stays at zero (terminal condition)
    
    # Loop backward over time
    for t in d.T:-1:1
        # Loop over brand states
        for b in 0:1
            # Loop over route usage states (permanent characteristic)
            for z in 1:d.zbin
                # Loop over mileage states
                for x in 1:d.xbin
                    
                    # Calculate row index in transition matrix
                    row = x + (z-1)*d.xbin
                    
                    # Compute v₁: value of continuing with current engine
                    # Flow utility + Expected future value (FV already contains β from previous iteration)
                    v1 = θ[1] + θ[2]*d.xval[x] + θ[3]*b + d.xtran[row,:]' * FV[(z-1)*d.xbin+1:z*d.xbin, b+1, t+1]
                    
                    # Compute v₀: value of replacing engine
                    # Mileage resets to 0 after replacement (FV already contains β)
                    v0 = d.xtran[1+(z-1)*d.xbin,:]' * FV[(z-1)*d.xbin+1:z*d.xbin, b+1, t+1]
                    
                    # Store future value using log-sum-exp formula
                    # FV stores β·EMAX (discounted expected max value)
                    FV[row, b+1, t] = d.β * (log(exp(v0) + exp(v1)) + Base.MathConstants.eulergamma)
                    
                end
            end
        end
    end
    return FV
end

################################################################################
# PART 4: DYNAMIC ESTIMATION - LOG LIKELIHOOD (Question 3d)
################################################################################

# log_likelihood_dynamic(θ, d)
#
# Compute log likelihood for the dynamic model using observed states only.
#
# KEY INSIGHT:
# - We solve for FV on the FULL state space (all possible states)
# - We evaluate likelihood only on OBSERVED states in the data
# - FV already contains β, so we DON'T multiply by β when using it
#
# From Problem Set equation (5):
# v₁(x_t,b) - v₀(x_t,b) = θ₀ + θ₁·x₁t + θ₂·b + 
#                          β·Σ log{exp(v₀,t+1) + exp(v₁,t+1)}·[f₁ - f₀]
#
# Since FV stores β·EMAX:
# v_diff = flow_diff + (xtran₁ - xtran₀)' * FV  [no extra β needed]

@views @inbounds function log_likelihood_dynamic(θ, d)
    # First, compute future values for all states given current θ
    FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
    compute_future_value!(FV, θ, d)
    
    # Now compute likelihood using only observed states
    loglike = 0.0
    
    # Loop over individual buses
    for i in 1:d.N
        
        # Pre-compute the row index for replacement (mileage = 0)
        row0 = (d.Zstate[i] - 1) * d.xbin + 1
        
        # Loop over time periods
        for t in 1:d.T
            
            # Get row index for current state (observed mileage and route usage)
            row1 = d.Xstate[i,t] + (d.Zstate[i] - 1) * d.xbin
            
            # Compute conditional value difference: v₁ - v₀
            
            # Part 1: Flow utility difference
            flow_diff = θ[1] + θ[2]*d.X[i,t] + θ[3]*d.B[i]
            
            # Part 2: Expected future value difference
            # Note: FV already contains β, so we don't multiply by β again
            ev_diff = (d.xtran[row1,:] .- d.xtran[row0,:])' * FV[row0:row0+d.xbin-1, d.B[i]+1, t+1]
            
            # Total conditional value difference
            v_diff = flow_diff + ev_diff
            
            # Add to log likelihood using efficient form
            loglike += (d.Y[i,t] == 1) * v_diff - log(1 + exp(v_diff))
            
        end
    end
    
    # Return NEGATIVE log likelihood (Optim minimizes)
    return -loglike
end

################################################################################
# PART 5: OPTIMIZATION WRAPPER (Question 3e-h)
################################################################################

"""
    estimate_dynamic_model(d; θ_start=nothing)

Estimate the dynamic discrete choice model using MLE.
"""
function estimate_dynamic_model(d; θ_start=nothing)
    println("="^70)
    println("DYNAMIC MODEL ESTIMATION")
    println("="^70)
    
    # Set starting values
    if isnothing(θ_start)
        θ_start = rand(3)  # Random start if nothing provided
        println("\nUsing random starting values: ", θ_start)
    else
        println("\nUsing provided starting values: ", θ_start)
    end
    
    # Define objective function
    objective = θ -> log_likelihood_dynamic(θ, d)
    
    # Time the likelihood evaluation
    println("\nTiming likelihood evaluation...")
    @time objective(θ_start)
    @time objective(θ_start)
    
    # Run optimization
    println("\nOptimizing (this may take several minutes)...")
    result = optimize(objective, θ_start, LBFGS(), 
                     Optim.Options(g_tol=1e-5, 
                                  iterations=100_000, 
                                  show_trace=true,
                                  show_every=10))
    
    # Display results
    println("\n" * "="^70)
    println("RESULTS")
    println("="^70)
    println("Parameter estimates:")
    println("  θ₀ (constant):     ", round(result.minimizer[1], digits=4))
    println("  θ₁ (mileage):      ", round(result.minimizer[2], digits=4))
    println("  θ₂ (brand):        ", round(result.minimizer[3], digits=4))
    println("\nLog likelihood:      ", round(-result.minimum, digits=2))
    println("Converged:           ", Optim.converged(result))
    println("Iterations:          ", Optim.iterations(result))
    println("="^70)
    
    return result
end

################################################################################
# MAIN EXECUTION WRAPPER (Question 3f)
################################################################################

"""
    main()

Main wrapper function that runs all estimation procedures.
"""
function main()
    println("\n" * "="^70)
    println("PROBLEM SET 5: BUS ENGINE REPLACEMENT MODEL")
    println("="^70)
    
    #---------------------------------------------------------------------------
    # Part 1: Static Estimation (Questions 1-2)
    #---------------------------------------------------------------------------
    println("\n" * "-"^70)
    println("PART 1: STATIC (MYOPIC) ESTIMATION")
    println("-"^70)
    
    # Load data
    println("\nLoading and reshaping data...")
    df_long = load_static_data()
    println("Observations: ", nrow(df_long))
    println("First few rows:")
    println(first(df_long, 5))
    
    # Estimate static model
    println("\nEstimating static logit model...")
    theta_static = estimate_static_model(df_long)
    
    # Extract coefficients to use as starting values for dynamic model
    θ_start_dynamic = coef(theta_static)
    println("\nStatic estimates to use as starting values:")
    println(θ_start_dynamic)
    
    #---------------------------------------------------------------------------
    # Part 2: Dynamic Estimation (Question 3)
    #---------------------------------------------------------------------------
    println("\n" * "-"^70)
    println("PART 2: DYNAMIC ESTIMATION")  
    println("-"^70)
    
    # Load data
    println("\nLoading dynamic data...")
    d = load_dynamic_data()
    println("Buses (N): ", d.N)
    println("Time periods (T): ", d.T)
    println("Mileage bins: ", d.xbin)
    println("Route usage bins: ", d.zbin)
    println("Total state space size: ", d.xbin * d.zbin)
    println("Discount factor (β): ", d.β)
    
    # Estimate dynamic model
    println("\nSetting up dynamic estimation...")
    result = estimate_dynamic_model(d, θ_start=θ_start_dynamic)
    
    println("\n" * "="^70)
    println("END OF PROBLEM SET")
    println("="^70)
    
    return result
end

################################################################################
# NOTES FOR STUDENTS
################################################################################
#
# WHAT YOU NEED TO UNDERSTAND:
#
# 1. THE ECONOMIC MODEL:
#    - Static: Zurcher is myopic (β=0), only cares about today
#    - Dynamic: Zurcher is forward-looking (β=0.9), considers future costs
#    - Decision: Replace engine (d=0) vs. Keep running (d=1)
#    - State: (mileage, route usage, brand, time)
#
# 2. BACKWARD RECURSION (the key algorithm):
#    - Start at t=T: No future, so FV[T+1] = 0
#    - Work backward: FV[t] depends on FV[t+1]
#    - Compute for ALL states (not just observed ones)
#    - This gives us E[V_{t+1}] for any state we might visit
#
# 3. WHY WE COMPUTE FV FOR ALL STATES:
#    - When at state (z=5, x=10, t=5), we need E[V(s',6)]
#    - The expectation is over all possible s' we might transition to
#    - We weight each possible s' by transition probability xtran
#    - So we need V(s',6) for ALL s' in the state space
#
# 4. THE TRANSITION MATRIX TRICK:
#    - xtran[row,:] gives P(x'|x, z, don't replace)
#    - xtran[row0,:] gives P(x'|0, z, replace)  
#    - In likelihood: (xtran[row1,:] - xtran[row0,:])' · FV
#    - This efficiently computes E[V|continue] - E[V|replace]
#
# 5. PERFORMANCE TIPS:
#    - @views avoids array copies
#    - @inbounds skips bounds checking
#    - Named tuples avoid long argument lists
#    - Together these cut runtime ~50%
#
# 6. DEBUGGING TIPS:
#    - Start with small test: compute FV for one θ value
#    - Check FV dimensions: (zbin*xbin, 2, T+1)
#    - Verify FV[T+1] = 0 (terminal condition)
#    - Check that v1 > v0 when mileage is low (replacing is costly)
#    - Print likelihood value - should be large negative number
#    - If likelihood = 0, you have a bug in the loops!
#
# Good luck! The hard part is understanding the algorithm, not coding it.
################################################################################