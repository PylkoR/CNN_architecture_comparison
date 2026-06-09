function im2col!(col::AbstractMatrix{Float32}, x::AbstractArray{Float32, 3}, k_w::Int, k_h::Int, pad::Int)
    W_x, H_x, C_in = size(x)
    W_y = W_x - k_w + 2pad + 1
    H_y = H_x - k_h + 2pad + 1

    col_idx = 1
    @inbounds for j in 1:H_y
        for i in 1:W_y
            row_idx = 1
            for c in 1:C_in
                c_offset = (c - 1) * W_x * H_x 
                
                for dj in 1:k_h
                    for di in 1:k_w
                        xi = i + di - 1 - pad
                        xj = j + dj - 1 - pad
                        
                        if 1 <= xi <= W_x && 1 <= xj <= H_x
                            linear_idx = xi + (xj - 1) * W_x + c_offset
                            col[row_idx, col_idx] = x[linear_idx]
                        else
                            col[row_idx, col_idx] = 0.0f0
                        end
                        row_idx += 1
                    end
                end
            end
            col_idx += 1
        end
    end
end

function col2im!(x_grad::AbstractArray{Float32, 3}, col::AbstractMatrix{Float32}, k_w::Int, k_h::Int, pad::Int)
    W_x, H_x, C_in = size(x_grad)
    W_y = W_x - k_w + 2pad + 1
    H_y = H_x - k_h + 2pad + 1

    col_idx = 1
    @inbounds for j in 1:H_y
        for i in 1:W_y
            row_idx = 1
            for c in 1:C_in
                for dj in 1:k_h
                    for di in 1:k_w
                        xi = i + di - 1 - pad
                        xj = j + dj - 1 - pad
                        if 1 <= xi <= W_x && 1 <= xj <= H_x
                            x_grad[xi, xj, c] += col[row_idx, col_idx]
                        end
                        row_idx += 1
                    end
                end
            end
            col_idx += 1
        end
    end
end

function glorot_uniform(fan_in::Int, fan_out::Int, dims...)
    limit = sqrt(6.0f0 / (fan_in + fan_out))
    return (rand(Float32, dims...) .- 0.5f0) .* Float32(2.0f0 * limit)
end
