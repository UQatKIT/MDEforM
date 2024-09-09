########################################################################################################
########################################################################################################
# Short examplatory code snippet for various parameter simulation studies of the MDE method for 
# multiscale diffusions. The code may has to be changed depending on the limit model of interest.
# The one given here considers an overdamped Langevin setting with linear drift.
########################################################################################################
########################################################################################################
# Jaroslav Borodavka, 08.09.2024

using MDEforM, Dates, Statistics

## small Monte Carlo parameter study ##

# different time horizons of the process
T_range = range(50, 100, 2)
T_length = length(T_range)
# number of realizations of a process
reps = 100

MDE_aver_values = Array{Float64}(undef, T_length)
MDE_stdev_values = Array{Float64}(undef, T_length)
MDE_loop_values = Array{Float64}(undef, reps)

time_stamp_start = Dates.format(now(), "H:MM:SS")
@info "∇ $(time_stamp_start) - Start of simulation runs."

for i in 1:T_length

    for j in 1:reps
        data = Langevin(1.0, 1.0/0.1, func_config=LDO(), α=2.0, σ=1.0, ϵ=0.1, T=T_range[i])[1]      # true parameter: ϑ=α*K(LDO()[3], σ)
        MDE_loop_values[j] = MDE(data, "Langevin", 1.0*K(LDO()[3], 1.0), 10.0)
    end
    
    MDE_aver_values[i] = Statistics.mean(MDE_loop_values)
    MDE_stdev_values[i] = Statistics.std(MDE_loop_values)

    time_stamp_loop = Dates.format(now(), "H:MM:SS")
    @info "∇ $(time_stamp_loop) - T = $(T_range[i]) complete."
end

@show MDE_aver_values