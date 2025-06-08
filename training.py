"""
Contains the training loop implementation, including logging, model evaluation, and saving the best-performing model.
Trained models are stored in the Models/ directory.
"""
import os
import numpy as np
import torch

from monai.networks.nets import UNet
from monai.data import Dataset, DataLoader
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss, HausdorffDTLoss
from monai.metrics import DiceMetric
from monai.handlers.utils import from_engine
from monai.networks.layers import Norm
from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from monai.networks import one_hot
from torch.utils.tensorboard import SummaryWriter

from image_preprocessing import creat_dataset_for_training
from config import CLASS_LABEL, SAVE_MODEL_FOLDER,LOG_DIR,TRAIN_EPOCHS,VALIDATION_INTERVAL

from util import create_folder

def get_model():
    model = UNet(
        spatial_dims=2,
        in_channels=1,         # grayscale
        out_channels=len(CLASS_LABEL.keys())+1,        # 4 classes: background, MID, VTB, EPI
        channels=(16, 32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )
    return model


def model_training():
    create_folder(SAVE_MODEL_FOLDER)
    create_folder(LOG_DIR)
    train_ds, val_ds = creat_dataset_for_training()

    print("Loaded Datasets")

    train_batch_size=2
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #1e-3
    #dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
    dice_metric = DiceMetric(include_background=False,reduction="mean_batch",get_not_nans=True)

    post_pred = AsDiscrete(argmax=True, to_onehot=len(CLASS_LABEL.keys())+1)
    post_label = AsDiscrete(to_onehot=len(CLASS_LABEL.keys())+1)
    #loss_fn = DiceLoss(softmax=True,to_onehot_y=True)
    dice_ce_loss = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        lambda_dice=0.5,
        lambda_ce=0.5,
        reduction="mean"
    )
    hausdorff_loss = HausdorffDTLoss(
        include_background=False, 
        to_onehot_y=True,
        softmax=True,
        reduction="mean")
    
    # Loss scheduler function
    def get_combined_loss(epoch, warmup_epochs=int(TRAIN_EPOCHS*0.2), max_weight=0.2):
        if epoch < warmup_epochs:
            return lambda outputs, labels: dice_ce_loss(outputs, labels)
        else:
            # Gradually increase Hausdorff weight
            progress = min((epoch - warmup_epochs) / (warmup_epochs * 2), 1.0)
            haus_weight = progress * max_weight
            dice_weight = 1.0 - haus_weight

            return lambda outputs, labels: (
                dice_weight * dice_ce_loss(outputs, labels) +
                haus_weight * hausdorff_loss(outputs, labels)
            )

    # def combined_loss(outputs, labels):
    #     #print("Loss output.shape : ",outputs.shape)
    #     #print("Loss labels.shape : ",labels.shape)
    #     #return 0.7 * dice_ce_loss(outputs, labels) + 0.3 * hausdorff_loss(outputs, labels)
    #     return dice_ce_loss(outputs, labels)
    
    writer = SummaryWriter(LOG_DIR)
    best_metric = -1
    best_metric_epoch = -1

    # loss_fn = DiceFocalLoss(
    #     to_onehot_y=True,
    #     softmax=True,
    #     include_background=False
    # )
    train_losses, val_dices = [], []

    for epoch in range(TRAIN_EPOCHS):
        print(f"\nEpoch {epoch+1}/{TRAIN_EPOCHS}")
        model.train()
        current_loss_fn = get_combined_loss(epoch)
        epoch_loss = 0

        for batch in train_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)


            loss = current_loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        writer.add_scalar("train/loss", avg_train_loss, epoch)
        print(f"Train Loss: {avg_train_loss:.4f}")

        # --- Validation ---
        if (epoch + 1) % VALIDATION_INTERVAL == 0:
            model.eval()
            epoch_val_loss=0
            with torch.no_grad():
                dice_metric.reset()
                for val_batch in val_loader:
                    val_inputs = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)

                    #print("Image val_inputs : ",val_inputs.shape)
                    val_outputs = model(val_inputs)

                    val_loss = current_loss_fn(val_outputs, val_labels)
                    epoch_val_loss += val_loss.item()

                    val_labels = val_labels.squeeze(1) #Remove Channel
                    val_outputs = val_outputs.squeeze(0)

                    val_outputs = post_pred(val_outputs)
                    val_labels = post_label(val_labels)

                    val_labels = val_labels.unsqueeze(0) #Add Batch
                    val_outputs = val_outputs.unsqueeze(0) #Add Batch

                    dice_metric(y_pred=val_outputs, y=val_labels)
                
                avg_val_loss = epoch_val_loss / len(val_loader)
                writer.add_scalar("val/loss", avg_val_loss, epoch)
                print(f"Val Loss: {avg_val_loss:.4f}")

                mean_dice_per_class,_ = dice_metric.aggregate()
                print("Per-Class Dice Scores:")
                class_names = list(CLASS_LABEL.keys())
                print(mean_dice_per_class)
                for i, score in enumerate(mean_dice_per_class):
                    print(f"{class_names[i]} mean Dice: {score.item():.4f}")
                    writer.add_scalar("val/"+str(class_names[i]), score.item(), epoch)
                metric = mean_dice_per_class.mean()
                print(f"Average Dice: {metric}")
                writer.add_scalar("val/mean_dice", metric, epoch)
                    #print(f"{class_names[i]} Dice: {score.cpu()[i].item():.4f}")

                save_path = SAVE_MODEL_FOLDER + f"unet_epoch_{epoch}.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_score": mean_dice_per_class.mean()
                }, save_path)

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch
                    best_model_path = SAVE_MODEL_FOLDER + "best_model.pth"

                    torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_score": mean_dice_per_class.mean()
                    }, best_model_path)
                    print(f"Saved best model at epoch {best_metric_epoch} with Dice: {best_metric:.4f}")
                    writer.add_text("Info/Model", f"Saved best model at epoch {best_metric_epoch} with Dice: {best_metric:.4f}", best_metric_epoch)

    writer.close()



#TESTING 
"""
Below is example on how to run this step.
"""
if __name__ == "__main__":
    model_training()