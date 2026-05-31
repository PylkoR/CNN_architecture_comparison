module auto_differ
    include("Utils.jl")
    include("Engine.jl")
    include("Operations.jl")
    include("Layers.jl")
    
    export Chain, Conv, MaxPool, Flatten, Dense, Dropout, SoftmaxCE,
           build_model, forward!, backward!, optimize!, zerograd!,
           IS_TRAINING, set_training!, GraphNode, build_graph
    end