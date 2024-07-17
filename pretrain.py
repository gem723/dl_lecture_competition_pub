import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.models import CLIPModel
from src.datasets import CLIPDataset, ThingsMEGDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def contrastive_loss(image_embeddings, brainwave_embeddings, temperature=0.07):
    logits = torch.matmul(image_embeddings, brainwave_embeddings.t()) / temperature
    labels = torch.arange(logits.shape[0]).to(logits.device)
    loss = nn.CrossEntropyLoss()(logits, labels) + nn.CrossEntropyLoss()(logits.t(), labels)
    return loss / 2

def calculate_accuracy(logits, labels):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

def pretrain(epochs=10, batch_size=32, lr=0.0001, num_workers=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # データの読み込み
    train_brainwave_data = ThingsMEGDataset("train", "data")
    val_brainwave_data = ThingsMEGDataset("val", "data")
    
    train_dataset = CLIPDataset("data/train_image_paths.txt", train_brainwave_data.X)
    val_dataset = CLIPDataset("data/val_image_paths.txt", val_brainwave_data.X)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # モデルの初期化
    model = CLIPModel(num_classes=1854, seq_len=train_brainwave_data.seq_len, in_channels=train_brainwave_data.num_channels, embedding_dim=512).to(device)

    # オプティマイザとスケジューラの設定
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 事前学習ループ
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for images, brainwaves in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images, brainwaves = images.to(device), brainwaves.to(device)
            
            image_embeddings, brainwave_embeddings = model(images, brainwaves)
            loss = contrastive_loss(image_embeddings, brainwave_embeddings)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += calculate_accuracy(torch.matmul(image_embeddings, brainwave_embeddings.t()), torch.arange(images.size(0)).to(device))

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for images, brainwaves in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                images, brainwaves = images.to(device), brainwaves.to(device)
                
                image_embeddings, brainwave_embeddings = model(images, brainwaves)
                loss = contrastive_loss(image_embeddings, brainwave_embeddings)
                
                val_loss += loss.item()
                val_acc += calculate_accuracy(torch.matmul(image_embeddings, brainwave_embeddings.t()), torch.arange(images.size(0)).to(device))

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | train loss: {train_loss:.3f} | train acc: {train_acc:.3f} | val loss: {val_loss:.3f} | val acc: {val_acc:.3f}")

    # 事前学習したモデルの保存
    torch.save(model.state_dict(), "pretrained_clip_model.pth")

if __name__ == "__main__":
    pretrain()