using Random, LinearAlgebra

# Tryb uczenia / ewaluacji dla DropOut
const IS_TRAINING = Ref(true)
set_training!(val::Bool) = IS_TRAINING[] = val

function glorot_uniform(dims::Int...)
    fan_in = length(dims) == 2 ? dims[2] : dims[1]*dims[2]*dims[3]
    fan_out = length(dims) == 2 ? dims[1] : dims[1]*dims[2]*dims[4]
    limit = sqrt(6.0f0 / (fan_in + fan_out))
    return rand(Float32, dims...) .* (2.0f0 * limit) .- limit
end

# Sprowadza obraz 3D do macierzy 2D dla szybkiego mnożenia
function im2col!(col::Matrix{Float32}, padded::Array{Float32, 3}, img::AbstractArray{Float32, 3}, k_w::Int, k_h::Int, pad::Int)
    w, h, c = size(img)
    out_w = w + 2pad - k_w + 1
    out_h = h + 2pad - k_h + 1
    
    fill!(padded, 0.0f0)
    padded[pad+1:pad+w, pad+1:pad+h, :] .= img
    
    col_idx = 1
    for y in 1:out_h
        for x in 1:out_w
            patch = view(padded, x:x+k_w-1, y:y+k_h-1, :)
            col[:, col_idx] .= vec(patch)
            col_idx += 1
        end
    end
end

# Zmienia macierz z powrotem na obraz z akumulacją gradientu
function col2im!(dX::Array{Float32, 3}, padded::Array{Float32, 3}, col::Matrix{Float32}, k_w::Int, k_h::Int, pad::Int)
    fill!(padded, 0.0f0)
    w, h, c = size(dX)
    out_w = w + 2pad - k_w + 1
    out_h = h + 2pad - k_h + 1
    
    col_idx = 1
    for y in 1:out_h
        for x in 1:out_w
            patch_flat = view(col, :, col_idx)
            idx = 1
            for cc in 1:c
                for yy in 1:k_h
                    for xx in 1:k_w
                        padded[x+xx-1, y+yy-1, cc] += patch_flat[idx]
                        idx += 1
                    end
                end
            end
            col_idx += 1
        end
    end
    dX .+= padded[pad+1:pad+w, pad+1:pad+h, :]
end