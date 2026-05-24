abstract type Operator end

struct Dense <: Operator
    insize::Int
    outsize::Int
    activation::Union{Nothing, Symbol}
end
Dense(pair::Pair{Int, Int}, act=nothing) = Dense(first(pair), last(pair), act)

struct Conv <: Operator
    k::Tuple{Int, Int}
    ch::Pair{Int, Int}
    pad::Int
end
Conv(k, ch; pad=0, bias=false) = Conv(k, ch, pad)

struct MaxPool <: Operator
    k::Tuple{Int, Int}
end

struct Flatten <: Operator end

struct Dropout <: Operator
    p::Float32
end

# Budowanie sieci z prealokacją cache
function (op::Dense)(x::GraphNode)
    W_data = glorot_uniform(op.outsize, op.insize)
    b_data = zeros(Float32, op.outsize)
    W = GraphNode{:weight}((), W_data, true)
    b = GraphNode{:weight}((), b_data, true)
    
    out_data = zeros(Float32, op.outsize)
    node = GraphNode{:dense}((W, b, x), out_data)
    
    if op.activation == :relu
        act_out = zeros(Float32, op.outsize)
        return GraphNode{:relu}((node,), act_out)
    end
    return node
end

function (op::Conv)(x::GraphNode)
    in_w, in_h, in_c = size(x.data)
    k_w, k_h = op.k
    out_c = op.ch[2]
    pad = op.pad
    
    out_w = in_w + 2pad - k_w + 1
    out_h = in_h + 2pad - k_h + 1
    
    W_data = glorot_uniform(k_w, k_h, in_c, out_c)
    W = GraphNode{:weight}((), W_data, true)
    out_data = zeros(Float32, out_w, out_h, out_c)
    
    # prealokacja cache
    padded_buf = zeros(Float32, in_w + 2pad, in_h + 2pad, in_c)
    col_buf = zeros(Float32, k_w * k_h * in_c, out_w * out_h)
    out_mat_buf = zeros(Float32, out_c, out_w * out_h)
    dY_mat_buf = zeros(Float32, out_c, out_w * out_h)
    dW_mat_buf = zeros(Float32, out_c, k_w * k_h * in_c)
    dX_col_buf = zeros(Float32, k_w * k_h * in_c, out_w * out_h)
    
    cache = (; pad, k_w, k_h, padded=padded_buf, col=col_buf, out_mat=out_mat_buf, 
               dY_mat=dY_mat_buf, dW_mat=dW_mat_buf, dX_col=dX_col_buf)
               
    node = GraphNode{:conv}((W, x), out_data, cache)
    return node
end

function (op::MaxPool)(x::GraphNode)
    in_w, in_h, in_c = size(x.data)
    k_w, k_h = op.k
    
    out_w, out_h = in_w ÷ k_w, in_h ÷ k_h
    out_data = zeros(Float32, out_w, out_h, in_c)
    
    indices = zeros(Int, out_w, out_h, in_c)
    
    cache = (; k_w, k_h, indices)
    
    return GraphNode{:maxpool}((x,), out_data, cache)
end

function (op::Flatten)(x::GraphNode)
    out_len = length(x.data)
    out_data = zeros(Float32, out_len)
    return GraphNode{:flatten}((x,), out_data)
end

function (op::Dropout)(x::GraphNode)
    p_node = GraphNode{:tensor}((), Float32[op.p])
    out_data = zeros(Float32, size(x.data))
    
    mask_buf = zeros(Float32, size(x.data))
    
    return GraphNode{:dropout}((x, p_node), out_data, mask_buf)
end

struct SoftmaxCE <: Operator end
function (op::SoftmaxCE)(logits::GraphNode, target::GraphNode)
    loss_data = zeros(Float32, 1)

    probs_buf = zeros(Float32, size(logits.data))
    
    # trzeci argument to cache
    return GraphNode{:softmax_ce}((logits, target), loss_data, probs_buf)
end

function Chain(layers...)
    return layers
end

function build_model(chain, input_shape, target_shape)
    # Wejściowe tensory dla obrazu i etykiety
    input_node = GraphNode{:tensor}((), zeros(Float32, input_shape...))
    target_node = GraphNode{:tensor}((), zeros(Float32, target_shape...))
    
    curr = input_node
    for layer in chain
        curr = layer(curr)
    end
    
    loss_node = SoftmaxCE()(curr, target_node)
    graph = build_graph(loss_node)
    
    # Posortowany graf oraz wskaźniki na wejścia, wyjścia i loss
    return graph, input_node, target_node, curr, loss_node
end