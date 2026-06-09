function predict(graph, input_node, output_node, x_data)
    forward!(graph, input_node => x_data, train_mode=false)
    return copy(output_node.data)
end

function zero_activations_grad!(order :: Vector{GraphNode})
    for node in order
        if !(node isa GraphWeight)
            node.grad .= 0
        end
    end
end

function train_step!(graph, input_node, target_node, x_batch, y_batch, eta)
    zerograd!(graph)
    batch_size = size(x_batch, 4)
    
    for i in 1:batch_size
        zero_activations_grad!(graph)
        
        x_single = @view x_batch[:, :, :, i]
        y_single = @view y_batch[:, i]
        
        forward!(graph, input_node => x_single, target_node => y_single, train_mode=true)
        backward!(graph)
    end
    
    optimize!(graph, eta / batch_size) 
end

function loss_and_accuracy_clean(graph, input_node, output_node, loss_node, target_node, data_loader)
    total_loss = 0.0f0
    correct = 0
    count = 0
    
    for (x_batch, y_batch) in data_loader
        batch_size = size(x_batch, 4)
        
        for i in 1:batch_size
            x_single = x_batch[:, :, :, i]
            y_single = y_batch[:, i]
            
            forward!(graph, input_node => x_single, target_node => y_single, train_mode=false)
            
            total_loss += loss_node.data[1]
            
            pred_idx = argmax(output_node.data)
            target_idx = argmax(y_single)
            if pred_idx == target_idx
                correct += 1
            end
            count += 1
        end
    end
    
    acc = round(100 * correct / count; digits=2)
    avg_loss = total_loss / count
    return (; loss=avg_loss, acc=acc)
end