include("../helper/MatrixMF.jl")
include("../helper/PoissonMaxFunctions.jl")
include("../helper/OrderStatsSampling.jl")
include("../helper/NegBinPMF.jl")
using Distributions
using LinearAlgebra

struct MaxNegBinMF <: MatrixMF
    N::Int64
    M::Int64
    K::Int64
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    D::Int64
    p::Float64
end

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

function evalulateLogLikelihood(model::MaxNegBinMF, state, data, info, row, col)
    # println("lol")
    Y = data["Y_NM"][row,col]
    mu = dot(state["U_NK"][row,:], state["V_KM"][:,col])
    if model.D == 1
        return logpdf(NegativeBinomial(mu,1-model.p), Y)
    # elseif model.D == model.j
    #     return logpmfMaxNegBin(Y, r, p, model.D)
    else
        return logpmfMaxNegBin(Y, mu, 1-model.p, model.D)
        #return logpmfOrderStatNegBin(Y, mu, model.p, model.D, model.D)
    end
end

# function evalulateLogLikelihood(model::MaxNegBinMF, state, data, info, row, col)
#     Y = data["Y_NM"][row,col]
#     mu = dot(state["U_NK"][row,:], state["V_KM"][:,col])
#     return logpdf(OrderStatistic(NegativeBinomial(mu, 1-model.p), model.D, model.D), Y)
# end

function sample_prior(model::MaxNegBinMF, info=nothing)
    U_NK = rand(Gamma(model.a, 1/model.b), model.N, model.K)
    V_KM = rand(Gamma(model.c, 1/model.d), model.K, model.M)
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM)
    return state
end

function forward_sample(model::MaxNegBinMF; state=nothing, info=nothing)
    if isnothing(state)
        state = sample_prior(model, info)
    end
    Mu_NM = state["U_NK"] * state["V_KM"]
    Y_NM = rand.(OrderStatistic.(NegativeBinomial.(Mu_NM, 1-model.p),model.D,model.D))

    data = Dict("Y_NM" => Y_NM)
    state = Dict("U_NK"=>U_NK, "V_KM"=>V_KM)
    return data, state 
end

function backward_sample(model::MaxNegBinMF, data, state, mask=nothing)
    #some housekeeping
    Y_NM = copy(data["Y_NM"])
    U_NK = copy(state["U_NK"])
    V_KM = copy(state["V_KM"])
    Z_NMK = zeros(model.N, model.M, model.K)
    P_K = zeros(model.K)
    Z1_NM = zeros(Int, model.N,model.M)
    Z2_NM = zeros(Int, model.N,model.M)
    Mu_NM = U_NK * V_KM
    #Loop over the non-zeros in Y_NM and allocate
    for n in 1:model.N
        for m in 1:model.M
            if !isnothing(mask)
                if mask[n,m] == 1
                    Y_NM[n,m] = rand(NegativeBinomial(Mu_NM[n,m], 1-model.p))
                end
            end
            if Y_NM[n, m] > 0
                
                #first sample sum of NegBin from NegBinMax
                Z1_NM[n,m] = sampleSumGivenOrderStatistic(Y_NM[n,m],model.D, model.D, NegativeBinomial(Mu_NM[n,m],1-model.p))
                #now sample poisson from sum of NegBin (which is NegBin)
                Z2_NM[n,m] = sampleCRT(Z1_NM[n,m],model.D*Mu_NM[n,m])
                #println(Y_NM[n, m], " ", Z1_NM[n,m], " ", Z2_NM[n,m])
                #now Z is a (certain kind of) Poisson so we can thin it
                P_K[:] = U_NK[n, :] .* V_KM[:, m]
                P_K[:] = P_K / sum(P_K)
                Z_NMK[n, m, :] = rand(Multinomial(Z2_NM[n, m], P_K))
            end
        end
    end

    for n in 1:model.N
        for k in 1:model.K
            post_shape = model.a + sum(Z_NMK[n, :, k])
            post_rate = model.b + model.D*log(1/(1-model.p))*sum(V_KM[k, :])
            U_NK[n, k] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end

    for m in 1:model.M
        for k in 1:K
            post_shape = model.c + sum(Z_NMK[:, m, k])
            post_rate = model.d + model.D*log(1/(1-model.p))*sum(U_NK[:, k])
            V_KM[k, m] = rand(Gamma(post_shape, 1/post_rate))[1]
        end
    end
    state = Dict("U_NK" => U_NK, "V_KM" => V_KM, "Z_NMK"=>Z_NMK, "Z1_NM"=>Z1_NM)
    return data, state
end