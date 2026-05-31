abstract type Operator end
const Chain = Vector{Operator}

# --- Chain ---
function chain(operators)
  function flatten(x::Tuple)
    y = Vector{Operator}()
    for v in x
      if v isa Tuple
        push!(y, v...)
      else
        push!(y, v)
      end
    end
    return y
  end

  result = Vector{Operator}()
  for operator in flatten(operators)
    push!(result, operator)
  end
  return result
end

function (ch::Chain)(x)
  node = x
  for op in ch
    node = op(node)
  end
  return node
end

# --- GraphNode ---
mutable struct GraphNode{OP, N}
  args :: NTuple{N, GraphNode}
  grad 
  data
  cache
end

const GraphWeight = GraphNode{:weight, 0}
const GraphTensor = GraphNode{:tensor, 0}

function GraphNode(data::T, trainable=false) where T
  if trainable
    return GraphNode{:weight, 0}((), zero(data), data, nothing)
  else
    return GraphNode{:tensor, 0}((), zero(data), data, nothing)
  end
end

function GraphNode(op::Symbol, args::Tuple, data::T; cache=nothing) where T
  N = length(args)
  grad = similar(data) 
  return GraphNode{op, N}(args, grad, data, cache)
end

function primal!(tensor::GraphTensor; train_mode=true)  end
function primal!(weight::GraphWeight; train_mode=true)  end
function adjoint!(::GraphTensor) end
function adjoint!(::GraphWeight) end

function graph(node)
  function visit!(node::GraphNode, visited, ordered)
    if node in visited
    else
      push!(visited, node)
      for arg in node.args
        visit!(arg, visited, ordered)
      end
      push!(ordered, node)
    end
    return nothing
  end
  ordered = Vector{GraphNode}()
  visited = Set{GraphNode}()
  visit!(node, visited, ordered)
  return ordered
end

function zerograd!(order :: Vector{GraphNode})
  for node in order
    node.grad .= 0
  end
end

function forward!(order::Vector{GraphNode}, pairs...; train_mode=true)
  for (tensor, data) in pairs
    tensor.data .= data
  end

  for node in order
    primal!(node, train_mode=train_mode)
  end
end

function backward!(order::Vector{GraphNode})
  seed = last(order)
  seed.grad .= 1

  for node in reverse(order)
    adjoint!(node)
  end
end