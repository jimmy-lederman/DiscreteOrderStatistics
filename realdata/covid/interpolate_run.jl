println("opended file")
flush(stdout)
# using Pkg
# Pkg.instantiate()
# Pkg.status()
# using Pkg
# Pkg.precompile()
include("/home/jlederman/DiscreteOrderStatistics/models/covid/covidsimple.jl")
using Dates
using CSV
using DataFrames
using Random
using JLD
println(Threads.nthreads())
println("imported packages")
flush(stdout)

#cumdf = Matrix(CSV.read("/Users/jimmy/Desktop/OrderStats/data/CTFL.csv",DataFrame))
#cumdf = Matrix(CSV.read("../data/CTFL.csv",DataFrame))
cumdf = Matrix(CSV.read("/home/jlederman/DiscreteOrderStatistics/data/usafull.csv",DataFrame))
days = cumdf[1,4:end]
state = cumdf[2:end,1]
fips = cumdf[2:end,2]
pops = log.(cumdf[2:end,3])
cumdf = cumdf[2:end,4:end]
data = Dict("Y_NM"=>cumdf)
N = size(cumdf)[1]
M = size(cumdf)[2]
Y_N1 = Int.(reshape(cumdf[:,1], :, 1))
info = Dict("Y0_N"=>Y_N1,"pop_N"=>pops,"days_M"=>Vector(days), "state_N"=>state)

maskSeed = parse(Int, ARGS[1])
chainSeed = parse(Int, ARGS[2])
D = parse(Int, ARGS[3])
j = parse(Int, ARGS[4])
K = parse(Int, ARGS[5])

Random.seed!(maskSeed)
mask_NM = rand(N,M) .< .025


# cumdf = Matrix(CSV.read("/home/jlederman/DiscreteOrderStatistics/data/CTFL.csv",DataFrame))
# a = .1
# b = 1
# c = 100
# d = .01
# starta = .01
# startb = 1
# e = 1000
# f = 1000
# g = .1
# h = 1
# scale_shape = .5
# scale_rate = 1
a = 1
b = 1
c = 100
d = .1
starta = .01
startb = 1
g = .5
h = 1
scale_shape = 2
scale_rate = 1

model = covidsimple(N,M,K,a,b,c,d,g,h,scale_shape,scale_rate,starta,startb,D,j)


@time samples = fit(model, data, initseed=chainSeed, nsamples = 100, nburnin=4000, nthin=10, mask=mask_NM,info=info,constantinit=Dict("V_KM"=>fill(1.0, K, M)))
# inforate = evaluateInfoRate(model,data,samples,mask=mask_NM, verbose=false,info=info)
results = [K,D,j,maskSeed,chainSeed]

samples = [Dict("eps"=>sample["eps"], "alpha"=>sample["alpha"], "V_KM"=>sample["V_KM"], "U_NK"=>sample["U_NK"]) for sample in samples]
folder = "/net/projects/schein-lab/jimmy/OrderStats/realdata/covid/medianD3/usa/interpolate_samples/"
save(folder*"/sample_maskSeed$(maskSeed)chainSeed$(chainSeed)D$(D)j$(j)K$(K).jld", "results", results, "samples", samples, "mask", mask_NM)
