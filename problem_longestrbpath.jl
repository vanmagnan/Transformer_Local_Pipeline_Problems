include("constants.jl")


using Random
using LinearAlgebra
using GraphPlot
using Graphs
using SimpleGraphAlgorithms
using SimpleGraphs
using SimpleGraphConverter

include("src/planarity.jl")


const N = 9

function minimize_longest_rb_path(A::Matrix{Int8})
    for i in 1:N
        A[i,i]=0
        for j in (i+1):N
            A[i,j]=A[j,i]
        end
    end
    V = shuffle([i for i in 1:N])
    for v in V
        if length(Set([A[v,j] for j in 1:N]))<N
            for j in V 
                if j!=v
                    if count(i->i==A[j,v],[A[v,j] for j in 1:N])>1
                        forbidden = union(Set([A[k,j] for k in 1:N if k!=v]),Set([A[k,v] for k in 1:N if k!=j]))
                        available = setdiff(Set([k for k in 1:2N]),forbidden)
                        A[j,v] = minimum(available)
                        A[v,j] = minimum(available)
                    end
                end
            end
        end
    end
    E = shuffle([[i,j] for i in 1:N, j in 1:N if j<i])
    l = reward(A)
    for e in E
        i = e[1]
        j = e[2]
        forbidden = union(Set([A[k,j] for k in 1:N if k!=i]),Set([A[i,k] for k in 1:N if k!=j]))
        used = Set([A[i,j] for i in 1:N, j in 1:N if j<i])
        new = minimum(setdiff(Set([k for k in 1:N^2]),used))
        available = union(setdiff(used, forbidden),Set([new]))
        for c in available
            current = A[i,j]
            A[i,j] = c
            A[j,i] = c
            if reward(A) < l
                break
            else 
                A[i,j] = current
                A[j,i] = current
                continue
            end
        end
    end
    return A
end

            
 
function convert_adjmat_to_string(adjmat::Matrix{Int8})::String
    entries = []

    # Collect entries from the upper diagonal of the matrix
    for i in 1:N-1
        for j in i+1:N
            push!(entries, string(adjmat[i, j]))
        end
        push!(entries, ",")
    end

    # Join all entries into a single string
    return join(entries)
end


function convert_adjmat_to_string(adjmat::Matrix{Int64})::String
    entries = []

    # Collect entries from the upper diagonal of the matrix
    for i in 1:N-1
        for j in i+1:N
            push!(entries, string(adjmat[i, j]))
        end
        push!(entries, ",")
    end

    # Join all entries into a single string
    return join(entries)
end




function greedy_search_from_startpoint(db, obj::String)::Vector{String}
    B=zeros(Int8,N,N)
    num_commas = count(c -> c == ',', obj)
    if num_commas == N - 1 #we've got the right size graph
        # Fill the upper triangular matrix
        index = 1
        for i in 1:N-1
            for j in i+1:N
                while obj[index] == ','
                    index += 1
                end
                #println(obj[index])
                B[i,j]=parse(Int8, obj[index])
                B[j,i]=B[i,j]
                index += 1
            end
        end
    end
    adjmat = minimize_longest_rb_path(B)
    return [convert_adjmat_to_string(adjmat)]
    # Now that we have 'adjmat', sample four random permutations
    permuted_adjmats = []
    for _ in 1:4
        perm = randperm(N)  # Generate a random permutation
        permuted_adjmat = adjmat[perm, perm]  # Apply the permutation to rows and columns
        push!(permuted_adjmats, permuted_adjmat)
    end
    return [convert_adjmat_to_string(permuted_adjmat) for permuted_adjmat in permuted_adjmats]
end



function reward(A::Matrix{Int8})::Int
    function longest_rb_path(A::Matrix{Int8}, startvtx::Int)::Int
        yettoexplore = [[[startvtx], []]]  # List of paths to explore: [path, colors]
        record = [[], []]  # Current longest rainbow path and its colors
        while !isempty(yettoexplore)
            current_path, current_colors = popfirst!(yettoexplore)  # Take the first path to explore
            for j in 1:N
                if A[current_path[end],j]!=0 && !(A[current_path[end],j] in current_colors) && !(j in current_path)
                    new_path = vcat(current_path, [j])
                    new_colors = vcat(current_colors, [A[current_path[end],j]])
                    yettoexplore = vcat(yettoexplore, [[new_path, new_colors]])
                    if length(new_path) > length(record[1])
                        record = [new_path, new_colors]
                    end
                end
            end
        end
        return length(record[1])  # Return the length of the longest rainbow path found
    end
    longestpath=0
    for vxstart in 1:N
        longestpath=max(longest_rb_path(A,vxstart),longestpath)
    end
    return longestpath
end

function reward_calc(obj::OBJ_TYPE)::REWARD_TYPE
    """
    Function to calculate the reward of a final construction
    (E.g. number of edges in a graph, etc)
    """
    A=zeros(Int8,N,N)
    # Fill the upper triangular matrix
    index = 1
    for i in 1:N-1
        for j in i+1:N
            while obj[index] == ','
                index += 1
            end
            #println(obj[index])
            A[i,j]=parse(Int8, obj[index])
            A[j,i]=A[i,j]
            index += 1
        end
    end
    return N-reward(A)
end

function empty_starting_point()::String
    """
    If there is no input file, the search starts always with this object
    (E.g. empty graph, all zeros matrix, etc)
    """
    adjmat = zeros(Int8, N, N)
    return convert_adjmat_to_string(adjmat)
end 
