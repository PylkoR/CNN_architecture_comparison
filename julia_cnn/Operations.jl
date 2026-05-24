# ================== DENSE  ==================
function primal!(y::GraphNode{:dense})
    W, b, x = y.args
    mul!(y.data, W.data, x.data)
    y.data .+= b.data
end

function adjoint!(y::GraphNode{:dense})
    W, b, x = y.args
    
    mul!(W.grad, y.grad, x.data', 1.0f0, 1.0f0)
    b.grad .+= y.grad
    mul!(x.grad, W.data', y.grad, 1.0f0, 1.0f0)

    if !isempty(x.args)
        mul!(x.grad, W.data', y.grad, 1.0f0, 1.0f0)
    end
end

# ================== RELU ==================
function primal!(y::GraphNode{:relu})
    x = y.args[1]
    y.data .= max.(0.0f0, x.data)
end

function adjoint!(y::GraphNode{:relu})
    x = y.args[1]
    x.grad .+= y.grad .* (x.data .> 0.0f0)
end

# ================== FLATTEN ==================
function primal!(y::GraphNode{:flatten})
    x = y.args[1]
    y.data .= vec(x.data)
end

function adjoint!(y::GraphNode{:flatten})
    x = y.args[1]
    x.grad .+= reshape(y.grad, size(x.data))
end

# ================== DROPOUT ==================
function primal!(y::GraphNode{:dropout})
    x, p_node = y.args
    p = p_node.data[1]
    mask = y.cache
    
    if IS_TRAINING[]
        scale = 1.0f0 / (1.0f0 - p)
        rand!(mask) 
        
        for i in eachindex(y.data)
            m_val = (mask[i] >= p) ? scale : 0.0f0
            mask[i] = m_val # Zachowuje wartość dla backward
            y.data[i] = x.data[i] * m_val
        end
    else
        y.data .= x.data
    end
end

function adjoint!(y::GraphNode{:dropout})
    x, _ = y.args
    mask = y.cache
    
    for i in eachindex(x.grad)
        x.grad[i] += y.grad[i] * mask[i]
    end
end

# ================== MAXPOOL  ==================
function primal!(y::GraphNode{:maxpool})
    x = y.args[1]
    x_data = x.data :: Array{Float32, 3}
    y_data = y.data :: Array{Float32, 3}
    k_w = y.cache.k_w :: Int
    k_h = y.cache.k_h :: Int
    indices = y.cache.indices :: Array{Int, 3}
    
    w, h, c = size(x_data)
    out_w, out_h = size(y_data, 1), size(y_data, 2)
    
    for ch in 1:c, j in 1:out_h, i in 1:out_w
        x_start = (i-1) * k_w + 1
        y_start = (j-1) * k_h + 1

        patch = view(x_data, x_start:x_start+k_w-1, y_start:y_start+k_h-1, ch)
        val, idx = findmax(patch)
        y_data[i, j, ch] = val
        
        indices[i, j, ch] = idx[1] + (idx[2]-1) * k_w 
    end
end

function adjoint!(y::GraphNode{:maxpool})
    x = y.args[1]
    
    x_grad = x.grad :: Array{Float32, 3}
    y_grad = y.grad :: Array{Float32, 3}
    k_w = y.cache.k_w :: Int
    k_h = y.cache.k_h :: Int
    indices = y.cache.indices :: Array{Int, 3}
 
    w, h, c = size(y_grad)
        
    for ch in 1:c, j in 1:h, i in 1:w
        x_start = (i-1) * k_w + 1
        y_start = (j-1) * k_h + 1
            
        idx = indices[i, j, ch]
        dx = (idx - 1) % k_w
        dy = (idx - 1) ÷ k_w
            
        x_grad[x_start + dx, y_start + dy, ch] += y_grad[i, j, ch]
    end
end

# ================== CONVOLUTION ==================
function primal!(y::GraphNode{:conv})
    W, x = y.args
    c = y.cache
    in_c, out_c = size(W.data, 3), size(W.data, 4)
    out_w, out_h, _ = size(y.data)
    
    # Transformacja obrazu do kolumn
    im2col!(c.col, c.padded, x.data, c.k_w, c.k_h, c.pad)
    
    # Mnożenie W * col bez alokacji
    W_mat = transpose(reshape(W.data, c.k_w * c.k_h * in_c, out_c))
    mul!(c.out_mat, W_mat, c.col)
    
    y.data .= reshape(transpose(c.out_mat), out_w, out_h, out_c)
end

function adjoint!(y::GraphNode{:conv})
    W, x = y.args
    c = y.cache
    in_c, out_c = size(W.data, 3), size(W.data, 4)
    out_w, out_h, _ = size(y.data)
    
    # Rozłożenie dY na płaską macierz
    for ch in 1:out_c
        c.dY_mat[ch, :] .= vec(y.grad[:, :, ch])
    end
    
    # Gradient po wagach: dW_mat = dY_mat * col^T
    mul!(c.dW_mat, c.dY_mat, transpose(c.col))
    W.grad .+= reshape(transpose(c.dW_mat), c.k_w, c.k_h, in_c, out_c)

    if !isempty(x.args)
        W_mat = transpose(reshape(W.data, c.k_w * c.k_h * in_c, out_c))
        mul!(c.dX_col, transpose(W_mat), c.dY_mat)
        col2im!(x.grad, c.padded, c.dX_col, c.k_w, c.k_h, c.pad)
    end
end

# ================== SOFTMAX + CROSS ENTROPY ==================
function primal!(y::GraphNode{:softmax_ce})
    logits, target = y.args
    probs = y.cache
    
    max_l = maximum(logits.data)
    probs .= exp.(logits.data .- max_l)
    probs ./= sum(probs)
    
    # Cross Entropy Loss
    y.data[1] = -sum(target.data .* log.(probs .+ 1f-8))
end

function adjoint!(y::GraphNode{:softmax_ce})
    logits, target = y.args
    probs = y.cache
    
    if !isempty(logits.args)
        logits.grad .+= y.grad[1] .* (probs .- target.data)
    end
end