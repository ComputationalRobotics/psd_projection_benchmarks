using MatrixDepot
using LinearAlgebra

import Random
Random.seed!(1234)

# Parse command line arguments
function parse_args(args)
    datasets = String[]
    instance_sizes = Int[]
    i = 1
    while i <= length(args)
        if args[i] == "--datasets"
            i += 1
            while i <= length(args) && !startswith(args[i], "--")
                push!(datasets, args[i])
                i += 1
            end
        elseif args[i] == "--instance_sizes"
            i += 1
            while i <= length(args) && !startswith(args[i], "--")
                push!(instance_sizes, parse(Int, args[i]))
                i += 1
            end
        else
            i += 1
        end
    end
    return datasets, instance_sizes
end

datasets, instance_sizes = parse_args(ARGS)

open("data/bin/meta.log", "w") do meta
    for dataset in datasets
        for n in instance_sizes
            if dataset == "squared"
                A = rand(n, n)
                A = (A + A') / 2
                
                F = qr(A)
                A = F.Q * Diagonal([1 / i^2 for i in 1:n]) * F.Q'
            elseif dataset == "cubed"
                A = rand(n, n)
                A = (A + A') / 2

                F = qr(A)
                A = F.Q * Diagonal([1 / i^3 for i in 1:n]) * F.Q'
            else
                A = matrixdepot(dataset, n)
            end

            # symmetrize A
            A = (A + A') / 2

            open("data/bin/$(dataset)-$(n).bin", "w") do io
                write(io, A)
            end

            # compute the maximum and minimum eigenvalues
            eigenvalues = eigen(A)
            lambda_max = maximum(eigenvalues.values)
            lambda_min = minimum(eigenvalues.values)

            # write the metadata
            write(meta, "dataset '$(dataset)' of size '$(n)'\n")
            write(meta, "\tlambda_max: $(lambda_max)\n")
            write(meta, "\tlambda_min: $(lambda_min)\n")
            write(meta, "\n")
        end
    end
end