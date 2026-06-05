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
  W_init = glorot_uniform(n, m, m, n)
  W = GraphNode(W_init, true)
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
  fan_out = k_w * k_h * C_out
  W_init = glorot_uniform(fan_in, fan_out, k_w, k_h, C_in, C_out)
  W = GraphNode(W_init, true)
  
  b = bias ? GraphNode(zeros(Float32, C_out), true) : nothing

  return ConvLayer(W, b, pad)
end

function (c::ConvLayer)(x)
  W_x, H_x, _ = size(x.data)
  k_w, k_h, C_in, C_out = size(c.W.data)
  
  W_y = W_x - k_w + 2 * c.pad + 1
  H_y = H_x - k_h + 2 * c.pad + 1
  
  col_size = (k_w * k_h * C_in, W_y * H_y)
  im2col_buffer = zeros(Float32, col_size)
  dcol_buffer   = zeros(Float32, col_size)
  
  cache = Dict{Symbol, Any}(
      :pad => c.pad, 
      :col => im2col_buffer, 
      :dcol => dcol_buffer
  )
  
  args = c.b !== nothing ? (c.W, x, c.b) : (c.W, x)
  return GraphNode(:conv, args, zeros(Float32, W_y, H_y, C_out), cache=cache)
end

function primal!(y::GraphNode{:conv}; train_mode=true)
  W = y.args[1]
  x = y.args[2]
  has_bias = length(y.args) == 3
  
  W_d = W.data::Array{Float32, 4}
  x_d = x.data::Array{Float32, 3}
  y_d = y.data::Array{Float32, 3}
  
  pad = y.cache[:pad]::Int
  col = y.cache[:col]::Matrix{Float32}
  
  k_w, k_h, C_in, C_out = size(W_d)
  W_y, H_y, _ = size(y_d)

  im2col!(col, x_d, k_w, k_h, pad)
  
  W_mat = reshape(W_d, :, C_out)
  Y_mat = reshape(y_d, :, C_out)
  
  mul!(Y_mat, transpose(col), W_mat)
  
  if has_bias
      b_d = y.args[3].data::Vector{Float32}
      for c_out in 1:C_out
          bias_val = b_d[c_out]
          @inbounds @simd for i in 1:(W_y * H_y)
              Y_mat[i, c_out] += bias_val
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
  y_g = y.grad::Array{Float32, 3}
  W_g = W.grad::Array{Float32, 4}
  x_g = x.grad::Array{Float32, 3}
  
  pad  = y.cache[:pad]::Int
  col  = y.cache[:col]::Matrix{Float32}
  dcol = y.cache[:dcol]::Matrix{Float32}
  
  k_w, k_h, C_in, C_out = size(W_d)
  
  W_mat = reshape(W_d, :, C_out)
  dW_mat = reshape(W_g, :, C_out)
  dY_mat = reshape(y_g, :, C_out)
  
  mul!(dW_mat, col, dY_mat, 1.0f0, 1.0f0)
  mul!(dcol, W_mat, transpose(dY_mat))
  
  col2im!(x_g, dcol, k_w, k_h, pad)
  
  if has_bias
      b_g = y.args[3].grad::Vector{Float32}
      for c_out in 1:C_out
          sum_val = 0.0f0
          @inbounds @simd for i in 1:size(dY_mat, 1)
              sum_val += dY_mat[i, c_out]
          end
          b_g[c_out] += sum_val
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

