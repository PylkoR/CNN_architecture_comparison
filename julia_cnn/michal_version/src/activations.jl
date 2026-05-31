# --- ReLU ---
struct ReLU <: Operator end
relu() = ReLU()

function (y::ReLU)(x)
  return GraphNode(:relu, (x,), zeros(Float32, size(x.data)...))
end

function primal!(y::GraphNode{:relu, 1}; train_mode=true)
  x, = y.args
  y.data .= max.(0, x.data)
  return nothing
end

function adjoint!(y::GraphNode{:relu, 1})
  x, = y.args
  x.grad .+= (x.data .> 0) .* y.grad
  return nothing
end

# --- SoftMax ---
struct SoftMax <: Operator end
softmax() = SoftMax()

function (s::SoftMax)(x)
  return GraphNode(:softmax, (x,), zeros(Float32, size(x.data)...))
end

function primal!(y::GraphNode{:softmax, 1}; train_mode=true)
  x, = y.args
  xmax = maximum(x.data)
  exps = exp.(x.data .- xmax)
  y.data .= exps ./ sum(exps)
  return nothing
end

function adjoint!(y::GraphNode{:softmax, 1})
    x, = y.args
    y_data = y.data
    y_grad = y.grad
    sum_grad_y = sum(y_grad .* y_data)
    x.grad .+= y_data .* (y_grad .- sum_grad_y)
    return nothing
end