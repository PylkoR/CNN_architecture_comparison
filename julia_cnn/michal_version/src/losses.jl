# --- CROSS ENTROPY ---
abstract type Loss end

struct CrossEntropy <: Loss end

function (E::CrossEntropy)(probs, targets)
    return GraphNode(:ce_simple, (probs, targets), zeros(Float32, 1))
end

function primal!(z::GraphNode{:ce_simple, 2}; train_mode=true)
    probs, targets = z.args
    eps = 1f-10
    val = 0.0f0
    @inbounds @simd for i in eachindex(probs.data)
        val += targets.data[i] * log(probs.data[i] + eps)
    end
    z.data[1] = -val
    return nothing
end

function adjoint!(z::GraphNode{:ce_simple, 2})
    probs, targets = z.args
    eps = 1f-10
    @. probs.grad -= (targets.data / (probs.data + eps)) * z.grad[1]
    return nothing
end