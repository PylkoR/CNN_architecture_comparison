using LinearAlgebra

# --- Flatten ---
struct Flatten <: Operator end

function (f::Flatten)(x)
  return GraphNode(:flatten, (x,), zeros(Float32, length(x.data)))
end

function primal!(y::GraphNode{:flatten, 1}; train_mode=true)
  x, = y.args
  y.data .= vec(x.data)
  return nothing
end

function adjoint!(y::GraphNode{:flatten, 1})
    x, = y.args
    x.grad .+= reshape(y.grad, size(x.data)...)
    return nothing
end

# --- Dense ---
struct Dense <: Operator
  W::GraphNode
  b::GraphNode
end

function dense(pair::Pair{Int64, Int64}, activation=nothing)
  n, m = pair
  W = GraphNode(randn(Float32, m, n) .* sqrt(2.0f0 / n), true)
  b = GraphNode(zeros(Float32, m), true)
  d = Dense(W, b)
  return activation === nothing ? d : tuple(d, activation())
end

function (l::Dense)(x)
  m = size(l.W.data, 1)
  mul = GraphNode(:mul, (l.W, x), zeros(Float32, m))
  add = GraphNode(:add, (mul, l.b), zeros(Float32, m))
  return add
end

function primal!(y::GraphNode{:mul, 2}; train_mode=true)
  W, x = y.args
  mul!(y.data, W.data, x.data)
  return nothing
end

function adjoint!(y::GraphNode{:mul, 2})
  W, x = y.args
  mul!(W.grad, y.grad, x.data', 1.0f0, 1.0f0) 
  mul!(x.grad, W.data', y.grad, 1.0f0, 1.0f0)
  return nothing
end

function primal!(z::GraphNode{:add, 2}; train_mode=true)
  x, b = z.args
  z.data .= x.data .+ b.data
  return nothing
end

function adjoint!(z::GraphNode{:add, 2})
  x, b = z.args
  x.grad .+= z.grad
  b.grad .+= z.grad
  return nothing
end

# --- Conv ---
struct ConvLayer <: Operator
  W::GraphNode
  b::Union{GraphNode, Nothing}
  pad::Int
end

function conv2d(kernel::Tuple{Int, Int}, channels::Pair{Int, Int}; pad::Int=0, bias::Bool=true)
  k_w, k_h = kernel
  C_in, C_out = channels
  
  fan_in = k_w * k_h * C_in
  W = GraphNode(randn(Float32, k_w, k_h, C_in, C_out) .* sqrt(2.0f0 / fan_in), true)
  
  b = bias ? GraphNode(zeros(Float32, C_out), true) : nothing

  return ConvLayer(W, b, pad)
end

function (c::ConvLayer)(x)
  W_x, H_x, _ = size(x.data)
  k_w, k_h, _, C_out = size(c.W.data)
  
  W_y = W_x - k_w + 2 * c.pad + 1
  H_y = H_x - k_h + 2 * c.pad + 1
  
  if c.b !== nothing
      return GraphNode(:conv, (c.W, x, c.b), zeros(Float32, W_y, H_y, C_out), cache=Dict(:pad => c.pad))
  else
      return GraphNode(:conv, (c.W, x), zeros(Float32, W_y, H_y, C_out), cache=Dict(:pad => c.pad))
  end
end

function primal!(y::GraphNode{:conv}; train_mode=true)
  W = y.args[1]
  x = y.args[2]
  has_bias = length(y.args) == 3
  
  W_d = W.data::Array{Float32, 4}
  x_d = x.data::Array{Float32, 3}
  y_d = y.data::Array{Float32, 3}
  
  pad = y.cache[:pad]::Int

  k_w, k_h, C_in, C_out = size(W_d)
  W_x, H_x, _ = size(x_d)
  W_y, H_y, _ = size(y_d)

  fill!(y_d, 0.0f0)

  @inbounds for c_out in 1:C_out, c_in in 1:C_in
      for j in 1:H_y, i in 1:W_y
          val = 0.0f0 :: Float32
          for dj in 1:k_h, di in 1:k_w
              xi = i + di - 1 - pad
              xj = j + dj - 1 - pad
              
              if 1 <= xi <= W_x && 1 <= xj <= H_x
                  val += x_d[xi, xj, c_in] * W_d[di, dj, c_in, c_out]
              end
          end
          y_d[i, j, c_out] += val
      end
  end
  
  if has_bias
      b_d = y.args[3].data::Vector{Float32}
      @inbounds for c_out in 1:C_out
          b_val = b_d[c_out]
          for j in 1:H_y, i in 1:W_y
              y_d[i, j, c_out] += b_val
          end
      end
  end
  return nothing
end

function adjoint!(y::GraphNode{:conv})
  W = y.args[1]
  x = y.args[2]
  has_bias = length(y.args) == 3
  
  W_d = W.data::Array{Float32, 4}
  x_d = x.data::Array{Float32, 3}
  y_g = y.grad::Array{Float32, 3}
  W_g = W.grad::Array{Float32, 4}
  x_g = x.grad::Array{Float32, 3}
  
  pad = y.cache[:pad]::Int

  k_w, k_h, C_in, C_out = size(W_d)
  W_x, H_x, _ = size(x_d)
  W_y, H_y, _ = size(y_g)

  @inbounds for c_out in 1:C_out, c_in in 1:C_in
      for j in 1:H_y, i in 1:W_y
          dy = y_g[i, j, c_out]
          for dj in 1:k_h, di in 1:k_w
              xi = i + di - 1 - pad
              xj = j + dj - 1 - pad
              
              if 1 <= xi <= W_x && 1 <= xj <= H_x
                  W_g[di, dj, c_in, c_out] += x_d[xi, xj, c_in] * dy
                  x_g[xi, xj, c_in] += W_d[di, dj, c_in, c_out] * dy
              end
          end
      end
  end
  
  if has_bias
      b_g = y.args[3].grad::Vector{Float32}
      @inbounds for c_out in 1:C_out
          sum_g = 0.0f0
          for j in 1:H_y, i in 1:W_y
              sum_g += y_g[i, j, c_out]
          end
          b_g[c_out] += sum_g
      end
  end
  return nothing
end

# --- MaxPool 2x2 ---
struct MaxPool <: Operator
    k_w::Int
    k_h::Int
end

MaxPool(kernel::Tuple{Int, Int}=(2, 2)) = MaxPool(kernel[1], kernel[2])

function (m::MaxPool)(x)
  W_x, H_x, C = size(x.data)
  W_y, H_y = W_x ÷ m.k_w, H_x ÷ m.k_h
  cache = zeros(Int, 2, W_y, H_y, C)
  return GraphNode(:maxpool, (x,), zeros(Float32, W_y, H_y, C), cache=Dict(:cache => cache, :k_w => m.k_w, :k_h => m.k_h))
end

function primal!(y::GraphNode{:maxpool, 1}; train_mode=true)
  x, = y.args
  x_d = x.data::Array{Float32, 3}
  y_d = y.data::Array{Float32, 3}
  
  idx_cache = y.cache[:cache]::Array{Int, 4}
  k_w = y.cache[:k_w]::Int
  k_h = y.cache[:k_h]::Int
  
  W_y, H_y, C = size(y_d)

  @inbounds for c in 1:C, j in 1:H_y, i in 1:W_y
      i_start = (i - 1) * k_w + 1
      j_start = (j - 1) * k_h + 1
      
      max_val = -Inf32
      max_i, max_j = i_start, j_start

      for dj in 0:(k_h-1), di in 0:(k_w-1)
          val = x_d[i_start+di, j_start+dj, c]
          if val > max_val
              max_val = val
              max_i, max_j = i_start+di, j_start+dj
          end
      end
      y_d[i, j, c] = max_val
      idx_cache[1, i, j, c] = max_i
      idx_cache[2, i, j, c] = max_j
  end
  return nothing
end

function adjoint!(y::GraphNode{:maxpool, 1})
  x, = y.args
  idx_cache = y.cache[:cache]::Array{Int, 4}
  y_g = y.grad::Array{Float32, 3}
  x_g = x.grad::Array{Float32, 3}

  W_y, H_y, C = size(y_g)

  @inbounds for c in 1:C, j in 1:H_y, i in 1:W_y
      max_i = idx_cache[1, i, j, c]
      max_j = idx_cache[2, i, j, c]
      x_g[max_i, max_j, c] += y_g[i, j, c]
  end
  return nothing
end

# --- Dropout ---
struct Dropout <: Operator
  p::Float64
end

function (d::Dropout)(x)
  return GraphNode(:dropout, (x,), zeros(Float32, size(x.data)...), cache=Dict{Symbol, Any}(:p => d.p))
end

function primal!(y::GraphNode{:dropout, 1}; train_mode=true)
  x, = y.args
  p = Float32(y.cache[:p])
  
  if train_mode
      mask = rand(Float32, size(x.data)...) .> p
      y.cache[:mask] = mask
      
      @. y.data = (x.data * mask) / (1.0f0 - p)
  else
      y.data .= x.data
  end
  return nothing
end

function adjoint!(y::GraphNode{:dropout, 1})
  x, = y.args
  if haskey(y.cache, :mask)
      p = Float32(y.cache[:p])
      mask = y.cache[:mask]
      
      @. x.grad += (y.grad * mask) / (1.0f0 - p)
  else
      x.grad .+= y.grad
  end
  return nothing
end

