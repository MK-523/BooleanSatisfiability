import torch
import torch.optim as optim

def train_rl(model, dataset, num_epochs=300, learning_rate=0.001):
    #initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        total_reward = 0

        for clauses in dataset:
            #ensure batch dimension
            if clauses.dim() == 1:
                clauses = clauses.unsqueeze(0)

            #forward pass: sample variables
            sampled_variables, log_probs = model(clauses)
            #compute policy loss and SAT reward
            loss, sat_reward = model.loss(clauses, sampled_variables, log_probs)

            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #accumulate statistics
            total_loss += loss.item()
            total_reward += sat_reward.item()

        #log every 100 epochs
        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / len(dataset)
            avg_reward = total_reward / len(dataset)
            print(f'Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Avg SAT Reward: {avg_reward:.4f}')

    return model
