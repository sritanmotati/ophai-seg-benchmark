import monai
import torch
import numpy as np
from tqdm import tqdm
from statistics import mean
from torch.optim import Adam
import matplotlib.pyplot as plt
from transformers import SamModel
import matplotlib.patches as patches
from transformers import SamProcessor
from IPython.display import clear_output
from torch.utils.data import DataLoader

from utils.data_utils import *
from utils.losses import *

class SAM:
    def __init__(self, shape, n_classes):
        self.shape = shape
        self.n_classes = n_classes
        model = SamModel.from_pretrained("facebook/sam-vit-base")
        for name, param in model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)
        self.model = model
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    def summary(self):
        self.model.summary()

    def train(self, train_gen, val_gen, train_steps, val_steps, save_path):
        train_images = [x[0] for x in train_gen]
        train_masks = [x[1] for x in train_gen]
        val_images = [x[0] for x in val_gen]
        val_masks = [x[1] for x in val_gen]
        
        train_dataset = SAMDataset(image_paths=train_images, mask_paths=train_masks, processor=self.processor)
        train_dataloader = DataLoader(train_dataset, batch_size=len(train_gen)//train_steps, shuffle=True)

        val_dataset = SAMDataset(image_paths=val_images, mask_paths=val_masks, processor=self.processor)
        val_dataloader = DataLoader(val_dataset, batch_size=len(val_gen)//val_steps, shuffle=True)
        
        num_epochs = 5

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        optimizer = Adam(self.model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
        
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        mean_train_losses, mean_val_losses = [], []

        best_val_loss = 100.0
        best_val_epoch = 0

        self.model.train()
        for epoch in range(num_epochs):
            epoch_losses = []
            for i, batch in enumerate(tqdm(train_dataloader)):
                outputs = self.model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)

                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().to(device)
                loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                epoch_losses.append(loss.item())

                if i % 50 == 0:
                    clear_output(wait=True)

                    fig, axs = plt.subplots(1, 3)
                    xmin, ymin, xmax, ymax = get_bounding_box(batch['ground_truth_mask'][0])
                    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')

                    axs[0].set_title('input image')
                    axs[0].imshow(batch["pixel_values"][0,1], cmap='gray')
                    axs[0].axis('off')

                    axs[1].set_title('ground truth mask')
                    axs[1].imshow(batch['ground_truth_mask'][0], cmap='copper')
                    axs[1].add_patch(rect)
                    axs[1].axis('off')

                    medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))

                    medsam_seg_prob = medsam_seg_prob.detach().cpu().numpy().squeeze()
                    medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

                    axs[2].set_title('predicted mask')
                    axs[2].imshow(medsam_seg, cmap='copper')
                    axs[2].axis('off')

                    plt.tight_layout()
                    plt.show()

            val_losses = []

            with torch.no_grad():
                for val_batch in tqdm(val_dataloader):

                    outputs = self.model(pixel_values=val_batch["pixel_values"].to(device),
                            input_boxes=val_batch["input_boxes"].to(device),
                            multimask_output=False)

                    predicted_val_masks = outputs.pred_masks.squeeze(1)
                    ground_truth_masks = batch["ground_truth_mask"].float().to(device)
                    val_loss = seg_loss(predicted_val_masks, ground_truth_masks.unsqueeze(1))

                    val_losses.append(val_loss.item())

                fig, axs = plt.subplots(1, 3)
                xmin, ymin, xmax, ymax = get_bounding_box(val_batch['ground_truth_mask'][0])
                rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')

                axs[0].set_title('input image')
                axs[0].imshow(val_batch["pixel_values"][0,1], cmap='gray')
                axs[0].axis('off')

                axs[1].set_title('ground truth mask')
                axs[1].imshow(val_batch['ground_truth_mask'][0], cmap='copper')
                axs[1].add_patch(rect)
                axs[1].axis('off')

                medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))

                medsam_seg_prob = medsam_seg_prob.detach().cpu().numpy().squeeze()
                medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

                axs[2].set_title('predicted mask')
                axs[2].imshow(medsam_seg, cmap='copper')
                axs[2].axis('off')

                plt.tight_layout()
                plt.show()

                if mean(val_losses) < best_val_loss:
                    torch.save(self.model.state_dict(), f"best_sam_training.pth")
                    print(f"Model Was Saved! Current Best val loss {best_val_loss}")
                    best_val_loss = mean(val_losses)
                    best_val_epoch = epoch
                else:
                    print("Model Was Not Saved!")

            print(f'EPOCH: {epoch}')
            print(f'Mean loss: {mean(epoch_losses)}')

            mean_train_losses.append(mean(epoch_losses))
            mean_val_losses.append(mean(val_losses))
        
        # callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10), tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)]
        # self.model.compile(optimizer='adam', loss=dice_coef_multi_loss)
        # return self.model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=100, validation_data=val_gen, validation_steps=val_steps, callbacks=callbacks).history
        
        return {'loss': mean_train_losses, 'val_loss': mean_val_losses}
        

    def predict(self, x):
        test_dataset = SAMDataset(image_paths=[i[0] for i in x], mask_paths=[i[1] for i in x], processor=self.processor)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        preds = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader):

                # forward pass
                outputs = self.model(pixel_values=batch["pixel_values"].cuda(),
                            input_boxes=batch["input_boxes"].cuda(),
                            multimask_output=False)

                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().cuda()
                
                medsam_seg_prob = predicted_masks.cpu().numpy().squeeze()
                preds.append(medsam_seg_prob)
            
        preds = np.array(preds)
        return preds

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path, cpu=False):
        if cpu:
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load(path))

    def get_model(self):
        return self.model