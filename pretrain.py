import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
from omegaconf import DictConfig
import hydra

from src.models import CLIPModel
from src.datasets import CombinedCLIPDataset, ThingsMEGDataset
from src.utils import set_seed

def contrastive_loss(image_embeddings, brainwave_embeddings, temperature=0.07):
    logits = torch.matmul(image_embeddings, brainwave_embeddings.t()).div_(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = nn.functional.cross_entropy(logits, labels) + nn.functional.cross_entropy(logits.t(), labels)
    return loss / 2

def calculate_accuracy(logits, labels):
    _, predicted = torch.max(logits, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

@torch.no_grad()
def extract_image_features(model, dataset, device, batch_size=64, num_workers=4):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    features = []
    
    for images, _, _, _ in tqdm(dataloader, desc="Extracting image features"):
        images = images.to(device, non_blocking=True)
        batch_features = model.image_encoder(images)
        batch_features = batch_features.squeeze(-1).squeeze(-1)
        features.append(batch_features.cpu())
    
    return torch.cat(features, dim=0)

@hydra.main(version_base=None, config_path="configs", config_name="pretrain_config")
def pretrain(args: DictConfig):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.use_wandb:
        wandb.init(project="MEG-classification-pretrain", config=args)
    
    # データの読み込みと前処理
    train_brainwave_data = ThingsMEGDataset("train", args.data_dir)
    val_brainwave_data = ThingsMEGDataset("val", args.data_dir)

    # 組み合わせたデータセットの作成
    train_dataset = CombinedCLIPDataset(os.path.join(args.data_dir, "train_image_paths.txt"), train_brainwave_data)
    val_dataset = CombinedCLIPDataset(os.path.join(args.data_dir, "val_image_paths.txt"), val_brainwave_data)

    # モデルの初期化
    model = CLIPModel(
        num_classes=train_brainwave_data.num_classes,
        seq_len=train_brainwave_data.seq_len,
        in_channels=train_brainwave_data.num_channels,
        out_feature_dim=args.feature_dim
    ).to(device)

    # extractedフォルダが存在しない場合は作成
    extracted_dir = 'extracted'
    os.makedirs(extracted_dir, exist_ok=True)

    # 画像特徴量の抽出
    train_features_path = os.path.join(extracted_dir, 'train_image_features.pt')
    val_features_path = os.path.join(extracted_dir, 'val_image_features.pt')

    if not os.path.exists(train_features_path) or not os.path.exists(val_features_path):
        print("Extracting image features...")
        train_features = extract_image_features(model, train_dataset, device, args.batch_size, args.num_workers)
        val_features = extract_image_features(model, val_dataset, device, args.batch_size, args.num_workers)
        torch.save(train_features, train_features_path)
        torch.save(val_features, val_features_path)
    else:
        print("Loading pre-extracted image features...")
        train_features = torch.load(train_features_path)
        val_features = torch.load(val_features_path)

    # TensorDatasetの作成
    train_dataset = TensorDataset(train_features, train_brainwave_data.X, train_brainwave_data.y)
    val_dataset = TensorDataset(val_features, val_brainwave_data.X, val_brainwave_data.y)

    # DataLoaderの準備
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # オプティマイザとスケジューラの設定
    optimizer = torch.optim.AdamW(model.brainwave_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # extractedフォルダが存在しない場合は作成
    pretrained_models_dir = 'pretrained_models'
    os.makedirs(pretrained_models_dir, exist_ok=True)

    # 学習ループ
    best_val_acc = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for image_features, brainwaves, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Training"):
            image_features, brainwaves, labels = image_features.to(device), brainwaves.to(device), labels.to(device)
            
            brainwave_embeddings = model.brainwave_encoder(brainwaves)
            
            loss = contrastive_loss(image_features, brainwave_embeddings)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += calculate_accuracy(torch.matmul(image_features, brainwave_embeddings.t()), 
                                            torch.arange(image_features.size(0), device=device))

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # 検証ループ
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for image_features, brainwaves, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Validation"):
                image_features, brainwaves, labels = image_features.to(device), brainwaves.to(device), labels.to(device)
                
                brainwave_embeddings = model.brainwave_encoder(brainwaves)
                loss = contrastive_loss(image_features, brainwave_embeddings)
                
                val_loss += loss.item()
                val_acc += calculate_accuracy(torch.matmul(image_features, brainwave_embeddings.t()), 
                                              torch.arange(image_features.size(0), device=device))

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": scheduler.get_last_lr()[0]
            })

        # 最良モデルの保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(pretrained_models_dir, 'best_pretrained_clip_model.pth'))

    # 最終モデルの保存
    torch.save(model.state_dict(), os.path.join(pretrained_models_dir, 'final_pretrained_clip_model.pth'))

    print("Pretraining completed. Best model saved as 'best_pretrained_clip_model.pth' and final model saved as 'final_pretrained_clip_model.pth'")

if __name__ == "__main__":
    pretrain()