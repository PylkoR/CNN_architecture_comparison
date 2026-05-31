function optimize!(graph, η)
  for node in graph
    if node isa GraphWeight
      node.data .-= η .* node.grad
    end
  end
end