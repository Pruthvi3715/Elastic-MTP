"""
Gated-LoRA MTP Adapter Training Pipeline.

Trains auxiliary prediction heads on top of frozen backbone model with gradient detachment:
L_MTP = L_NTP + sum(lambda_i * CE(P_i, y_{t+1+i}))
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
from src.config import ElasticMTPConfig
from src.mtp_glora_adapter import MTPGLoRAModule

CHECKPOINT_DIR = os.path.join(ElasticMTPConfig.BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class SyntheticTrainerLM(nn.Module):
    def __init__(self, vocab_size=50257, hidden_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.backbone = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        emb = self.embedding(input_ids) # (batch, seq, hidden_dim)
        h = self.backbone(emb)
        logits = self.head(h)
        return logits, h

def run_adapter_training(epochs: int = 5, batch_size: int = 4, seq_len: int = 32, lr: float = 1e-3):
    print("=" * 65)
    print("Gated-LoRA MTP Adapter Training Pipeline")
    print("=" * 65)
    
    device = ElasticMTPConfig.DEVICE
    print(f"[Training] Operating on device: {device}")
    
    # 1. Initialize Base Backbone & Freeze Base Parameters
    vocab_size = 50257
    hidden_dim = 256
    num_aux_heads = 3
    
    base_model = SyntheticTrainerLM(vocab_size=vocab_size, hidden_dim=hidden_dim).to(device)
    for param in base_model.parameters():
        param.requires_grad = False # Freeze backbone!
        
    print("[Training] Base model parameters frozen successfully (0 backbone grads).")

    # 2. Attach Gated-LoRA Adapter Module
    adapter_module = MTPGLoRAModule(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_aux_heads=num_aux_heads,
        rank=8,
        lambda_0=0.3,
        gamma=0.8
    ).to(device)

    # 3. Optimizer for Adapter Parameters Only
    optimizer = optim.AdamW(adapter_module.parameters(), lr=lr, weight_decay=0.01)
    
    train_history = []
    start_time = time.time()
    
    # 4. Training Loop
    base_model.eval()
    adapter_module.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Synthetic batch input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # Primary forward pass
        primary_logits, hidden_states = base_model(input_ids) # (batch, seq, vocab), (batch, seq, hidden)
        
        # Primary NTP loss at t+1
        ntp_targets = input_ids[:, 1:]
        ntp_logits = primary_logits[:, :-1, :]
        primary_loss = torch.nn.functional.cross_entropy(
            ntp_logits.reshape(-1, vocab_size), 
            ntp_targets.reshape(-1)
        )
        
        # Compute Auxiliary Head Predictions with Gradient Detachment
        aux_logits_list = []
        aux_targets_list = []
        
        # Extract z_t at final position
        z_t = hidden_states[:, -1, :] # (batch, hidden)
        prev_emb = base_model.embedding(input_ids[:, -1]) # (batch, hidden)
        
        for k in range(num_aux_heads):
            head = adapter_module.aux_heads[k]
            aux_logits = head(z_t, prev_emb) # z_t is detached inside head!
            aux_logits_list.append(aux_logits)
            
            # Dummy future targets
            aux_target = torch.randint(0, vocab_size, (batch_size,), device=device)
            aux_targets_list.append(aux_target)
            
        # Composite Loss Calculation
        composite_loss, loss_dict = adapter_module.compute_composite_mtp_loss(
            primary_loss=primary_loss,
            aux_logits_list=aux_logits_list,
            target_tokens_list=aux_targets_list
        )
        
        # Backward Pass & Gradient Step
        composite_loss.backward()
        optimizer.step()
        
        train_history.append(loss_dict)
        print(f"[Epoch {epoch+1}/{epochs}] L_NTP: {loss_dict['primary_loss']:.4f} | L_MTP Total: {loss_dict['composite_total_loss']:.4f}")

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"Training Complete in {elapsed_time:.2f}s!")
    
    # 5. Save Adapter Checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "mtp_glora_adapter.pt")
    torch.save(adapter_module.state_dict(), checkpoint_path)
    print(f"Saved Gated-LoRA Adapter Checkpoint to: {checkpoint_path}")
    print("=" * 65)
    
    return checkpoint_path, train_history

if __name__ == "__main__":
    run_adapter_training()
