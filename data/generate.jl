using MatrixDepot

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

for dataset in datasets
    for n in instance_sizes
        A = matrixdepot(dataset, n)

        # symmetrize A
        A = (A + A') / 2

        open("data/bin/$(dataset)-$(n).bin", "w") do io
            write(io, size(A,1))
            write(io, size(A,2))
            write(io, A)
        end
    end
end