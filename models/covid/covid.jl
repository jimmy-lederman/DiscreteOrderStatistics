include("../../helper/MatrixMF.jl")
include("../../helper/PoissonMedianFunctions.jl")
using Distributions
using LogExpFunctions
using LinearAlgebra
# using Base.Threads
using SpecialFunctions

function sampleCRT(Y,R)
    if Y == 0
        return 0
    elseif Y == 1
        probs = [1]
    else
        probs = vcat([1],[R/(R+i-1) for i in 2:Y])
    end
    return sum(rand.(Bernoulli.(probs)))
end


struct covid <: MatrixMF
    N::Int64
    M::Int64
    T::Int64
    S::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    starta::Float64
    startb::Float64
    e::Float64
    f::Float64
    g::Float64 
    h::Float64
    scaleshape::Float64
    scalerate::Float64
    D::Int64
    j::Int64
end

# function evalulateLogLikelihood(model::OrderStatisticPoissonTimeDayMF, state, data, info, row, col)
#     @assert col != 1
#     Ylast = data["Y_NM"][row,col-1]
#     Y = data["Y_NM"][row,col]
#     pop = info["pop_N"][row]
#     day = info["days_M"][col]
#     s = info["state_N"][row]
#     eps = state["eps"]
#     alpha = state["alpha"]
#     mu = sum(state["U_NK"][row,:] .* state["V_KM"][:,col] .* state["R_KTS"][:,day,s])
#     return logpmfOrderStatPoisson(Y,Ylast+alpha*pop*mu+pop*eps,model.D,model.j)
# end

function forecast(model::covid, state, data, info, Ti)
    lastgamma = state["V_KM"][:,end]
    forecastGamma_KTi = zeros(model.K, Ti)
    for i in 1:Ti
        forecastGamma_KTi[:,i] = rand.(Gamma.(model.c*lastgamma,1/model.c))
        lastgamma = copy(forecastGamma_KTi[:,i])
    end 
    Ylast = data["Y_NM"][:,end]
    #generate new dats of the week
    last_day_of_week = info["days_M"][end]
    days = [mod1(last_day_of_week + i - 1, 7) for i in 1:Ti]

    R_NKTi = permutedims(state["R_KTS"][:,days,info["state_N"]],(3,1,2))
    temp_KTiN = permutedims(state["U_NK"] .* R_NKTi,(2,3,1))

    mu_NTi = permutedims(dropdims(sum(forecastGamma_KTi .* temp_KTiN, dims=1),dims=1),(2,1))

    Y_NTi = zeros(model.N,Ti)
    for i in 1:Ti
        mu = Ylast .+ info["pop_N"].*(state["eps"] .+ state["alpha"]*mu_NTi[:,i])
        Y_NTi[:,i] = rand.(OrderStatistic.(Poisson.(mu), model.D, model.j))
        Ylast = copy(Y_NTi[:,i])
    end
    return Y_NTi
end


function sample_prior(model::covid,info=nothing,constantinit=nothing)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    pass = false
    if !isnothing(constantinit)
        pass = ("V_KM" in keys(constantinit))
    end
    V_KM = zeros(model.K, model.M)
    if !pass
        @views for m in 1:model.M
            @views for k in 1:model.K
                if m == 1
                    V_KM[k,m] = rand(Gamma(model.starta,1/model.startb))
                else
                    V_KM[k,m] = rand(Gamma(model.c*V_KM[k,m-1]+model.d,1/model.c))
                end
            end
        end
    end
    R_KTS = rand(Gamma(model.e, 1/model.f), model.K, model.T, model.S)
    eps = rand(Gamma(model.g, 1/model.h))
    alpha = rand(Gamma(model.scaleshape, 1/model.scalerate))
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "R_KTS" => R_KTS, "eps"=>eps, "alpha" => alpha,
     "Y_N1"=>info["Y_N1"], "pop_N"=>info["pop_N"], "days_M"=>info["days_M"], "state_N"=>info["state_N"])
    return state
end

function forward_sample(model::covid; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    #Mu_NM = state["U_NK"] * state["V_KM"]
    eps = state["eps"]
    Y_N1 = state["Y_N1"]
    pop_N = state["pop_N"]
    days_M = state["days_M"]
    state_N = state["state_N"]
    alpha = state["alpha"]
    Y_NMKplus1 = zeros(Int, model.N, model.M, model.K + 2, model.D)
    Y_NM = zeros(Int, model.N, model.M)
    for n in 1:model.N
        for m in 1:model.M
            if m > 1
                day = days_M[m]
                s = state_N[n]
                mu = Y_NM[n,m-1] + pop_N[n]*eps + alpha * pop_N[n] * sum(state["R_KTS"][:,day,s] .* state["U_NK"][n,:] .* state["V_KM"][:,m])
                Y_NM[n,m] = rand(OrderStatistic(Poisson(mu), model.D, model.j))
            end
        end
    end

    # println(size(Y_NM), size(Y_NMD), size(Y_N1))
    Y_NM[:,1] = Y_N1[:,1]
    data = Dict("Y_NM" => Y_NM)
    state = Dict("U_NK" =>  state["U_NK"], "V_KM" => state["V_KM"], "R_KTS" => state["R_KTS"], "eps" => eps, "alpha" => alpha,
     "pop_N"=>info["pop_N"], "Y_N1"=>info["Y_N1"], "days_M"=>info["days_M"], "state_N"=>info["state_N"], "Y_NMKplus1"=>Y_NMKplus1)
    return data, state 
end

function predict(model::covid, state, info, n, m, Ylast)
    pop_N = info["pop_N"]
    days_M = info["days_M"]
    state_N = info["state_N"]
    eps = state["eps"]
    alpha = state["alpha"]
    day = days_M[m]
    s = state_N[n]
    pop = pop_N[n]
    return rand(OrderStatistic(Poisson(Ylast + pop*eps + alpha * pop * sum(state["R_KTS"][:,day,s] .* state["U_NK"][n,:] .* state["V_KM"][:,m])), model.D, model.j))
end

function predict_x(model::covid, state, info, n, mstart, Ystart, x)
    result = zeros(x)
    Ylast = Ystart
    m = mstart + 1
    for i in 1:length(result)
        result[i] = predict(model, state, info, n, m, Ylast)
        m += 1
        Ylast = result[i]
    end
    return result
end




function backward_sample(model::covid, data, state, mask=nothing, skipupdate=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    R_KTS = copy(state["R_KTS"])
    eps = copy(state["eps"])
    alpha = copy(state["alpha"])

    Y_N1 = copy(state["Y_N1"]) #unused because redundant in backwards step
    pop_N = copy(state["pop_N"])
    days_M = copy(state["days_M"])
    state_N = copy(state["state_N"])

    #mylock = ReentrantLock()

    alphacontribution_NM = zeros(model.N, model.M)
    Y_NMKplus1 = zeros(model.N, model.M, model.K + 2)
    # Loop over the non-zeros in Y_NM and allocate
    @views for idx in 1:(model.N * model.M)
        m = model.M - div(idx - 1, model.N)
        n = mod(idx - 1, model.N) + 1 
        if m == 1
            continue
        end
        @assert m != 1
        Ylast = Y_NM[n,m-1]
        pop = pop_N[n]
        day = days_M[m]
        s = state_N[n]
        probvec  = pop * R_KTS[:,day,s] .* U_NK[n,:] .* V_KM[:,m]
        alphacontribution_NM[n,m] = mu = sum(probvec) 

        if !isnothing(mask)
            if mask[n,m] == 1   
                Y_NM[n,m] = rand(OrderStatistic(Poisson(Ylast + pop*eps + alpha*mu), model.D, model.j))
                #Y_NM[n,m] = rand(OrderStatistic(Poisson(Ylast + alpha*mu), model.D, model.D))
            end
        end

        Z = sampleSumGivenOrderStatistic(Y_NM[n, m], model.D, model.j, Poisson(Ylast + alpha*mu + pop*eps))
        
        probvec = vcat(alpha*probvec, Ylast, pop*eps)

        Y_NMKplus1[n, m, :] = rand(Multinomial(Z, probvec / sum(probvec)))
    end

    #update alpha (scale)
    post_shape = model.scaleshape + sum(Y_NMKplus1[:,2:end,1:model.K])
    post_rate = model.scalerate + model.D*sum(alphacontribution_NM)
    alpha = rand(Gamma(post_shape, 1/post_rate))

    #update time factors
    Y_MK = dropdims(sum(Y_NMKplus1,dims=1),dims=1)[:,1:model.K]
    R_NKM = permutedims(R_KTS[:,days_M,state_N], (3,1,2))
    C_KM = model.D*alpha * dropdims(sum(pop_N .* U_NK .* R_NKM, dims=1), dims=1)

    l_KM = zeros(model.K,model.M+1)
    q_KM = zeros(model.K,model.M+1)
    #backward pass
    @views for m in model.M:-1:2
        @views for k in 1:model.K
            q_KM[k,m] = log(1 + (C_KM[k,m]/model.c) + q_KM[k,m+1])
            temp = sampleCRT(Y_MK[m,k] + l_KM[k,m+1], model.c*V_KM[k,m-1] + model.d)
            l_KM[k,m] = rand(Binomial(temp, model.c*V_KM[k,m-1]/(model.c*V_KM[k,m-1] + model.d)))
        end 
    end
    @assert sum(l_KM[:,model.M+1]) == 0 && sum(q_KM[:,model.M+1]) == 0
    #forward pass
    @views for m in 1:model.M
        @views for k in 1:model.K
            if m == 1
                V_KM[k,m] = rand(Gamma(model.starta + l_KM[k,m+1], 1/(model.startb + model.c*q_KM[k,m+1])))
            else
                V_KM[k,m] = rand(Gamma(model.d + model.c*V_KM[k,m-1] + Y_MK[m,k] + l_KM[k,m+1], 1/(model.c + C_KM[k,m] + model.c*q_KM[k,m+1])))
            end 
        end 
    end 

    #update day,state factors
    if isnothing(skipupdate) ||  !("R_KTS" in skipupdate)
        @views for t in 1:model.T
            indices = days_M .== t #filter to days of week
            indices[1] = 0 # do not include first day
            Vsubset_KB = V_KM[:,indices]
            @views for s in 1:model.S
                indices2 = state_N .== s #filter to state
                popsubset_A = pop_N[indices2]
                Usubset_AK = U_NK[indices2, :]
                Ysubset_ABK = Y_NMKplus1[indices2, indices, :]
                @views for k in 1:model.K
                    post_shape = model.e + sum(Ysubset_ABK[:,:,k])   
                    post_rate = model.f + model.D * alpha * dot(popsubset_A, Usubset_AK[:,k]) * sum(Vsubset_KB[k,:])
                    R_KTS[k,t,s] = rand(Gamma(post_shape, 1/post_rate))
                end
            end
        end
    end

    #update county factors
    Y_NK = dropdims(sum(Y_NMKplus1,dims=2),dims=2)[:,1:model.K]
    R_KMN = R_KTS[:,days_M[2:end],state_N]
    C_KN = model.D * alpha * dropdims(sum(V_KM[:,2:end] .* R_KMN, dims=2), dims=2)
    @views for n in 1:model.N
        pop =  pop_N[n]
        s = state_N[n]
        @views for k in 1:model.K
            post_shape = model.a + Y_NK[n,k]
            post_rate = model.b + pop * C_KN[k,n]
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))
        end
    end

    #update noise term
    post_shape = model.g + sum(Y_NMKplus1[:,:,model.K+2])
    post_rate = model.h + model.D*(model.M-1)*sum(pop_N)
    eps = rand(Gamma(post_shape, 1/post_rate))



    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "Y_N1"=>Y_N1, "R_KTS"=>R_KTS, "eps" => eps, "alpha" => alpha,
     "Y_NMKplus1"=>Y_NMKplus1, "pop_N"=>pop_N, "days_M"=>days_M, "state_N" => state_N)
    return data, state
end