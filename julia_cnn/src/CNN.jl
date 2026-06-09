module CNN

using LinearAlgebra
using Random
using Statistics: mean

# Engine Components
export GraphNode, GraphWeight, GraphTensor, Operator, Chain, chain
export forward!, backward!, zerograd!, graph, primal!, adjoint!

# Layers
export Dense, dense, ConvLayer, conv2d, MaxPool, Flatten, Dropout

# Losses & Optimize
export Loss, CrossEntropy, optimize!

# Training Utils
export train_step!, predict, loss_and_accuracy_clean

# Activation Functions
export ReLU, relu, SoftMax, softmax

# Utils
export im2col!, col2im!

include("engine.jl")
include("activations.jl")
include("layers.jl")
include("losses.jl")
include("optimizers.jl")
include("training.jl")
include("utils.jl")

end