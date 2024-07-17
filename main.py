import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier, ImprovedConvClassifier, CLIPModel
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    # モデルの初期化
    model = CLIPModel(
        num_classes=train_set.num_classes,
        seq_len=train_set.seq_len,
        in_channels=train_set.num_channels
    ).to(args.device)

    # 事前学習したモデルの読み込み
    pretrained_dict = torch.load("pretrained_models/final_pretrained_clip_model.pth")
    model_dict = model.state_dict()

    # 事前学習済みの重みを部分的にロード
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 're_classifier' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print("Pretrained weights loaded successfully")

    # # re_classifierの初期化
    # torch.nn.init.xavier_uniform_(model.re_classifier.weight)
    # torch.nn.init.zeros_(model.re_classifier.bias)

    # print("Re-classifier initialized")

    # image_encoderのパラメータを凍結
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    # ------------------
    #     Optimizer
    # ------------------
    # オプティマイザの設定
    optimizer = torch.optim.AdamW([
        {'params': model.brainwave_encoder.parameters(), 'lr': args.lr},  # 低い学習率
        {'params': model.re_classifier.parameters(), 'lr': args.lr}  # 通常の学習率
    ], weight_decay=args.weight_decay)

    # 学習率スケジューラーを追加
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)

            # デバイスの一貫性を確認
            # print(f"Device - model: {next(model.parameters()).device}, X: {X.device}, y: {y.device}")

            # ラベルの範囲を確認
            # print(f"y min: {y.min()}, y max: {y.max()}, num_classes: {model.brainwave_encoder.conv_classifier.head[-1].out_features}")

            y_pred = model(X)

            # モデルの出力とラベルの形状を確認
            # print(f"y_pred shape: {y_pred.shape}, y shape: {y.shape}")
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())
        
        # 学習率のスケジューリング
        scheduler.step()
        
        # 現在の学習率を表示
        lr_brainwave = optimizer.param_groups[0]['lr']
        lr_re_classifier = optimizer.param_groups[1]['lr']
        print(f"Current learning rates - Brainwave Encoder: {lr_brainwave:.6f}, Re-classifier: {lr_re_classifier:.6f}")

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)
            
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
