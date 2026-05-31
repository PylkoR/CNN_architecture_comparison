mutable struct GraphNode{OP, N, T <: AbstractArray{Float32}, C}
    args::NTuple{N, GraphNode}
    data::T
    grad::T
    cache::C
    trainable::Bool

    function GraphNode{OP}(args::Tuple, data::T, trainable::Bool=false) where {OP, T}
        N = length(args)
        grad = zeros(Float32, size(data))
        new{OP, N, T, Nothing}(args, data, grad, nothing, trainable)
    end
    
    function GraphNode{OP}(args::Tuple, data::T, cache::C, trainable::Bool=false) where {OP, T, C}
        N = length(args)
        grad = zeros(Float32, size(data))
        new{OP, N, T, C}(args, data, grad, cache, trainable)
    end
end

# Sortowanie topologiczne grafu
function build_graph(seed::GraphNode)
    ordered = Vector{GraphNode}()
    visited = Set{GraphNode}()
    function visit!(node::GraphNode)
        if !(node in visited)
            push!(visited, node)
            for arg in node.args
                visit!(arg)
            end
            push!(ordered, node)
        end
    end
    visit!(seed)
    return ordered
end

function zerograd!(order::Vector{GraphNode})
    for node in order
        fill!(node.grad, 0.0f0)
    end
end

primal!(node::GraphNode) = nothing
adjoint!(node::GraphNode) = nothing

function forward!(order::Vector{GraphNode}, pairs...)
    for (tensor, val) in pairs
        tensor.data .= val
    end
    for node in order
        primal!(node)
    end
end

function backward!(order::Vector{GraphNode})
    seed = order[end]
    fill!(seed.grad, 1.0f0)
    for node in reverse(order)
        adjoint!(node)
    end
end

function optimize!(graph::Vector{GraphNode}, η::Float32)
    for node in graph
        if node.trainable
            node.data .-= η .* node.grad
        end
    end
end

import Base: show
show(io::IO, x::GraphNode{OP, N, T, C}) where {OP, N, T, C} = 
    print(io, "layer ", OP, " with ", N, " arg(s)")