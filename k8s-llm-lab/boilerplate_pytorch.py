"""
RAW PYTORCH BOILERPLATE CODE (For ML Researchers / Custom Pipelines)
====================================================================
While Hugging Face's `Trainer` handles all the math for you, sometimes 
senior MLOps engineers write "Raw PyTorch Training Loops" if they need 
highly customized gradient updates or custom hardware routing.

This file shows the raw PyTorch calculus engine in action.
"""

import torch
from torch.optim import AdamW

def raw_pytorch_training_loop(model, train_dataloader, epochs=3):
    """
    This is the exact math loop that Hugging Face hides from you.
    """
    
    # 1. Boilerplate: Setup Optimizer
    # The Optimizer decides HOW to update the weights (AdamW is standard for LLMs)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Put the model into "Training Mode" (enables dropout, etc.)
    model.train()
    
    print("Starting Raw PyTorch Training Loop...")
    
    # 2. Boilerplate: The Epoch Loop
    for epoch in range(epochs):
        total_loss = 0
        
        # 3. Boilerplate: The Batch Loop
        for batch in train_dataloader:
            
            # Step A: Clear old gradients
            optimizer.zero_grad()
            
            # Step B: Forward Pass (Predict the next word)
            # The model automatically calculates the "Loss" (how wrong it was)
            outputs = model(**batch)
            loss = outputs.loss
            
            # Step C: Backward Pass (The Calculus)
            # This triggers PyTorch's Autograd engine to calculate the 
            # derivatives/gradients for millions of parameters.
            loss.backward()
            
            # Step D: Update Weights
            # The optimizer adjusts the weights based on the calculated gradients
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")
        
    print("Training Complete!")
    return model

