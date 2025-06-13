import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gc
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import random
import torchvision.transforms.functional as F
import threading
import gradio as gr
import pandas as pd
import sys
import time
import glob

class AutoencoderData(Dataset):
    def __init__(self, X, Y=None, augment_data=False, fram_gen_mode=False):
        if fram_gen_mode:
            assert X.shape == Y.shape
        self.X = X
        if fram_gen_mode:
            self.Y = Y
        self.fram_gen_mode = fram_gen_mode
        self.augment_data = augment_data
        if augment_data:
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.fram_gen_mode:
            y = self.Y[idx]
        if self.augment_data and random.random() > 0.5:
            for t in self.transforms.transforms:
                if isinstance(t, transforms.RandomHorizontalFlip):
                    if random.random() < t.p:
                        x = F.hflip(x)
                        if self.fram_gen_mode:
                            y = F.hflip(y)
                elif isinstance(t, transforms.ColorJitter):
                    brightness, contrast, saturation, hue = t.brightness, t.contrast, t.saturation, t.hue
                    fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = transforms.ColorJitter.get_params(
                        brightness, contrast, saturation, hue
                    )
                    for fn_id in fn_idx:
                        if fn_id == 0 and brightness_factor is not None:
                            x = F.adjust_brightness(x, brightness_factor)
                            if self.fram_gen_mode:
                                y = F.adjust_brightness(y, brightness_factor)
                        elif fn_id == 1 and contrast_factor is not None:
                            x = F.adjust_contrast(x, contrast_factor)
                            if self.fram_gen_mode:
                                y = F.adjust_contrast(y, contrast_factor)
                        elif fn_id == 2 and saturation_factor is not None:
                            x = F.adjust_saturation(x, saturation_factor)
                            if self.fram_gen_mode:
                                y = F.adjust_saturation(y, saturation_factor)
                        elif fn_id == 3 and hue_factor is not None:
                            x = F.adjust_hue(x, hue_factor)
                            if self.fram_gen_mode:
                                y = F.adjust_hue(y, hue_factor)
        if self.fram_gen_mode:
            return x, y
        else:
            return x, x
              # (encoded_dim,)

from sc_model import NextFramePredictor, NextFramePredictor64


def plot_outputs(frame_predictor, epoch, test_dataset, device, n=3, return_full = True):
    plt.figure(figsize=(16, 9))
    plt.title(f"Epoch: {epoch}", loc="left")
    input_images = []
    output_images = []
    wanted_images = []
    for i in range(n):
        img, wanted = test_dataset[i]
        img = img.unsqueeze(0).to(device)
        wanted = wanted.unsqueeze(0).to(device)
        frame_predictor.eval()
        with torch.no_grad():
            rec_img = frame_predictor(img)
        input_images.append(img.cpu().squeeze().permute(1, 2, 0).numpy())
        output_images.append(rec_img.cpu().squeeze().permute(1, 2, 0).numpy())
        wanted_images.append(wanted.cpu().squeeze().permute(1, 2, 0).numpy())
    input_image = np.concatenate(input_images, axis=1)
    output_image = np.concatenate(output_images, axis=1)
    wanted_image = np.concatenate(wanted_images, axis=1)

    plt.savefig("./latest.png")
    plt.close()
    if return_full:
        full_img = np.concatenate([input_image, output_image, wanted_image], axis=0) * 255
        full_img = full_img.astype(np.uint8)
        full_img = Image.fromarray(full_img)
        full_img.save("latest_image.jpg")
        del input_images, output_images, wanted_images, input_image, output_image, wanted_image
        gc.collect()
        return full_img
    else:
        del input_images, output_images, wanted
        gc.collect()
        return input_image, output_image, wanted_image


def train_loop(model, train_loader, val_loader, test_data, device, epochs, lr, weight_decay, out_path, step_count=0, epoch_count=0, best_val=float('inf'), return_full=True, callback_epoch=None, callback_end=None, callback_break=None, callback_new_best=None):
    
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    for epoch in range(1, epochs+1):
        if callback_break != None:
            if callback_break():
                if device.type == 'cuda':
                    torch.cuda.empty_cache()  
                return
        model.train()
        total_tr = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            total_tr += loss.item() * xb.size(0)
            step_count += 1
        avg_tr = total_tr / len(train_loader.dataset)
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                total_val += criterion(pred, yb).item() * xb.size(0)
        avg_val = total_val / len(val_loader.dataset)
        epoch_count += 1
        print(f"Epoch {epoch:3d}/{epochs}  train_loss={avg_tr:.6f}  val_loss={avg_val:.6f}")
        if callback_epoch is not None:
            examples = plot_outputs(model, f"{epoch}/{epochs}", test_data, device, 3, return_full)
            if isinstance(callback_epoch, list):
                for cb in callback_epoch:
                    cb(examples, epoch, avg_tr, avg_val, epoch_count, step_count)
            else:
                callback_epoch(examples, epoch, avg_tr, avg_val, epoch_count, step_count)
        if avg_val < best_val:
            best_val = avg_val
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            model.save(out_path)
            if callback_new_best is not None:
                if isinstance(callback_new_best, list):
                    for cb in callback_new_best:
                        cb(best_val)
                else:
                    callback_new_best(best_val)
    if callback_end is not None:
        if isinstance(callback_end, list):
            for cb in callback_end:
                cb()
        else:
            callback_end()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

def load_npy(X_path, y_path, shuffle=True, normalize=True, frame_gen_mode = False):
    X = np.load(X_path, mmap_mode='r').astype(np.float32)
    if frame_gen_mode:
        Y = np.load(y_path, mmap_mode='r').astype(np.float32)
        assert X.shape == Y.shape
    if normalize:
        X = X / 255.0
        if frame_gen_mode:
            Y = Y / 255.0
    if shuffle:
        keys = np.random.permutation(X.shape[0])
        X = X[keys]
        if frame_gen_mode:
            Y = Y[keys]
        del keys
        gc.collect()
    X = torch.from_numpy(X).permute(0, 3, 1, 2)
    if frame_gen_mode:
        Y = torch.from_numpy(Y).permute(0, 3, 1, 2)
        return X, Y
    
    else:
        return X, None

def load_npys(X_paths, y_paths, shuffle=True, normalize=True, balance_amt=-1, balance_ratio = -1, frame_gen_mode=False):
    assert len(X_paths) == len(y_paths)
    balance_dataset = balance_amt > -1 or balance_ratio != -1
    X_list = []
    if frame_gen_mode:
        y_list = []
    avg_ratio = []
    for x_path, y_path in zip(X_paths, y_paths):
        _X, _y = load_npy(x_path, y_path, shuffle, normalize, frame_gen_mode)
        if balance_dataset:
            if balance_ratio > 0:
                balance_amt = int(_X.shape[0] * balance_ratio)

            y_len = balance_amt / _X.shape[0]
            avg_ratio.append(y_len)

            _X = _X[:balance_amt]

            if frame_gen_mode:
                # y_len = int(_y.shape[0] * y_len)
                _y = _y[:balance_amt]
    
        X_list.append(_X)
        if frame_gen_mode:
            y_list.append(_y)
    avg_ratio = np.mean(avg_ratio)
    X = torch.cat(X_list, dim=0)
    if frame_gen_mode:
        y = torch.cat(y_list, dim=0)
    
        del X_list, y_list
    else:
        del X_list
    gc.collect()
    if shuffle:
        keys = torch.randperm(X.shape[0])
        X = X[keys]
        if frame_gen_mode:
            y = y[keys]
        del keys
        gc.collect()
    print("Training Data Shape X:", X.shape)
    if frame_gen_mode:
        print("Training Data Shape y:", y.shape)
    if frame_gen_mode:
        return X, y, avg_ratio
    else:
        return X, None, avg_ratio

def load_dataset(X_train, y_train, X_val, y_val, X_test, y_test, batch_size, shuffle, balance_amt, frame_gen_mode=False):
    print("Loading training data...")
    X, y, b_ratio = load_npys(X_train, y_train, shuffle, True, balance_amt, -1, frame_gen_mode)
    train_ds = FrameDataset(X, y, True, frame_gen_mode)
    
    del X, y
    gc.collect()
    print("Loading validation data...")
    X_val_data, y_val_data, _ = load_npys(X_val, y_val, shuffle, True, -1, b_ratio, frame_gen_mode)
    val_ds = FrameDataset(X_val_data, y_val_data, True, frame_gen_mode)
    del X_val_data, y_val_data
    gc.collect()
    print("Loading testing data...")
    X_test_data, y_test_data, _ = load_npys(X_test, y_test, shuffle, True, -1, b_ratio, frame_gen_mode)
    test_ds = FrameDataset(X_test_data, y_test_data, False, frame_gen_mode)
    del X_test_data, y_test_data
    gc.collect()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    del train_ds, val_ds
    gc.collect()
    return train_loader, val_loader, test_ds

# def shuffle_dataset(X_train, y_train, X_val, y_val, X_test, y_test):
#     X_train = np.array(X_train)
#     y_train = np.array(y_train)
#     X_val = np.array(X_val)
#     y_val = np.array(y_val)
#     X_test = np.array(X_test)
#     y_test = np.array(y_test)
#     keys = np.random.permutation(X_train.shape[0])
#     X_train = X_train[keys]
#     y_train = y_train[keys]
#     keys = np.random.permutation(X_val.shape[0])
#     X_val = X_val[keys]
#     y_val = y_val[keys]
#     keys = np.random.permutation(X_test.shape[0])
#     X_test = X_test[keys]
#     y_test = y_test[keys]
#     del keys
#     gc.collect()
#     return list(X_train), list(y_train), list(X_val), list(y_val), list(X_test), list(y_test)

def shuffle_dataset(X_train, y_train, X_val, y_val, X_test, y_test, use_random, dataset_window = 1):
    data_df = pd.DataFrame({"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val, "X_test": X_test, "y_test": y_test})
    data_df["dataset"] = data_df["X_train"].apply(lambda xt: os.path.split(xt)[-2])

    data_df = data_df.sample(frac=1)
    unique = data_df["dataset"].unique()

    if dataset_window > len(unique):
        dataset_window = len(unique) - 1
    
    np.random.shuffle(unique)
    unique_len = len(unique)
    previous_ds = [""]

    shuffled_df = pd.DataFrame(columns = data_df.columns)

    data_df.reset_index(inplace=True, drop=True)

    for index in range(len(data_df)):
        previous_ds = previous_ds[-dataset_window:]
        try:
            row = data_df[data_df["dataset"] == unique[index % unique_len] if not use_random else data_df["dataset"].apply(lambda ds: ds not in previous_ds)].sample(n=1)
        except:
            row = data_df.sample(n=1)
        
        for i, r in row.iterrows():
            shuffled_df.loc[len(shuffled_df)] = r
            data_df = data_df.drop(i)
    X_train, y_train, X_val, y_val, X_test, y_test = shuffled_df["X_train"].values, shuffled_df["y_train"].values, shuffled_df["X_val"].values, shuffled_df["y_val"].values, shuffled_df["X_test"].values, shuffled_df["y_test"].values
    
    del data_df, shuffled_df
    gc.collect()

    return X_train, y_train, X_val, y_val, X_test, y_test


def balance_dataset(X_train, y_train, X_val, y_val, X_test, y_test, min_amount = -1):
    data_df = pd.DataFrame({"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val, "X_test": X_test, "y_test": y_test})
    data_df["dataset"] = data_df["X_train"].apply(lambda xt: os.path.split(xt)[-2])

    dataset_names = data_df["dataset"].unique()
    dataset_amts = [len(data_df[data_df["dataset"] == d_name]) for d_name in dataset_names]
    minimum_amt = min(dataset_amts) if min_amount == -1 else min_amount
    print("Dataset Amounts:", dataset_amts)
    if min_amount > min(dataset_amts):
        minimum_amt = min(dataset_amts)
    
    new_df = pd.concat([data_df[data_df["dataset"] == d_name].sample(n=minimum_amt) for d_name in dataset_names])
    X_train, y_train, X_val, y_val, X_test, y_test = new_df["X_train"].values, new_df["y_train"], new_df["X_val"], new_df["y_val"], new_df["X_test"], new_df["y_test"]
    del new_df, dataset_amts, dataset_names, data_df
    gc.collect()

    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == '__main__':
    p = argparse.ArgumentParser("Next-Frame Generator")

    p.add_argument('--x_path', type=str, required=True, help='.npy of shape (n, 240, 240, 3) or (n, 64, 64, 3)')
    p.add_argument('--y_path', type=str, required=True, help='.npy of shape (n, 240, 240, 3) or (n, 64, 64, 3)')
    p.add_argument("--x_val", type=str, required=True, help='.npy of shape (n, 240, 240, 3) or (n, 64, 64, 3)')
    p.add_argument("--y_val", type=str, required=True, help='.npy of shape (n, 240, 240, 3) or (n, 64, 64, 3)')
    p.add_argument("--x_test", type=str, required=True, help=".npy of shape (n, 240, 240, 3) or (n, 64, 64, 3)")
    p.add_argument("--y_test", type=str, required=True, help=".npy of shape (n, 240, 240, 3) or (n, 64, 64, 3)")
    p.add_argument('--encoded_dim', type=int, default=768, help='size of bottleneck')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument("--true_epochs", type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--balance_amt", type=int, default=-1)
    p.add_argument('--model_out', type=str, default='./next_frame.pt')
    p.add_argument("--model_in", type=str, default="./next_frame.pt")
    p.add_argument('--no_cuda', action='store_true')
    p.add_argument("--dataset_steps", type=int, default=1)
    p.add_argument("--dataset_window", type=int, default=1)
    p.add_argument("--train_small", action="store_true")

    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    print("Device:", device)

    X_train = glob.glob(args.x_path) if "*" in args.x_path else [args.x_path]
    y_train = glob.glob(args.y_path) if "*" in args.y_path else [args.y_path]
    X_val = glob.glob(args.x_val) if "*" in args.x_val else [args.x_val]
    y_val = glob.glob(args.y_val) if "*" in args.y_val else [args.y_val]
    X_test = glob.glob(args.x_test) if "*" in args.x_test else [args.x_test]
    y_test = glob.glob(args.y_test) if "*" in args.y_test else [args.y_test]

    X_train, y_train, X_val, y_val, X_test, y_test = shuffle_dataset(X_train, y_train, X_val, y_val, X_test, y_test)

    if os.path.exists(args.model_in):
        model = NextFramePredictor.load(args.model_in, args.encoded_dim) if not args.train_small else NextFramePredictor64.load(args.model_in, args.encoded_dim)
    else:
        model = NextFramePredictor(args.encoded_dim) if not args.train_small else NextFramePredictor64(args.encoded_dim)

    print(model)

    train_df = pd.DataFrame(columns=["Epochs", "Loss"])
    val_df = pd.DataFrame(columns=["Epochs", "Loss"])
    example_img = np.zeros((480, 640, 3))
    current_epoch = 0
    current_true_epoch = 0
    epochs = args.epochs
    true_epochs = args.true_epochs
    training = False
    top_val = float("inf")

    def epoch_cb(img, epoch, avg_train_loss, avg_val_loss, epoch_count, step_count):
        global train_df, val_df, example_img, current_epoch, current_true_epoch
        current_epoch = epoch
        train_df.loc[len(train_df)] = {"Epochs": len(train_df) + 1, "Loss": avg_train_loss}
        val_df.loc[len(val_df)] = {"Epochs": len(val_df) + 1, "Loss": avg_val_loss}
        if img.width == 64 * 3:
            img = img.resize((240 * 3, 240 * 3))
        example_img = np.array(img)
        del img
        gc.collect()

    def new_top_cb(best_val_loss):
        global top_val
        top_val = best_val_loss

    def train():
        global current_true_epoch
        for tp in range(args.true_epochs):
            for index in range(0, len(X_train), args.dataset_steps):
                second_ind = (index + args.dataset_window) % len(X_train)
                if second_ind > index:
                    train_loader, val_loader, val_ds = load_dataset(X_train[index:second_ind], 
                                                                    y_train[index:second_ind], 
                                                                    X_val[index:second_ind], 
                                                                    y_val[index:second_ind],
                                                                    X_test[index:second_ind],
                                                                    y_test[index:second_ind], args.batch_size, args.shuffle, args.balance_amt)
                elif second_ind == 0:
                    train_loader, val_loader, val_ds = load_dataset(X_train[index:], 
                                                                    y_train[index:], 
                                                                    X_val[index:], 
                                                                    y_val[index:],
                                                                    X_test[index:],
                                                                    y_test[index:], args.batch_size, args.shuffle, args.balance_amt)
                else:
                    train_loader, val_loader, val_ds = load_dataset(X_train[index:] + X_train[:second_ind], 
                                                                    y_train[index:] + y_train[:second_ind], 
                                                                    X_val[index:] + X_val[:second_ind], 
                                                                    y_val[index:] + y_val[:second_ind],
                                                                    X_test[index:] + X_test[:second_ind],
                                                                    y_test[index:] + y_test[:second_ind], args.batch_size, args.shuffle, args.balance_amt)
                train_loop(model, train_loader, val_loader, val_ds, device,
                           epochs=epochs,
                           lr=args.lr,
                           weight_decay=args.weight_decay,
                           out_path=args.model_out, best_val=top_val, callback_epoch=[epoch_cb], callback_new_best=[new_top_cb])
                del train_loader, val_loader, val_ds
                gc.collect()
            current_true_epoch = tp + 1

    t = threading.Thread(target=train)

    def start_training():
        global t, train_df, val_df, example_img, training, current_epoch, epochs, current_true_epoch, true_epochs
        if not training and not t.is_alive():
            t.start()
            training = True
        while t.is_alive():
            time.sleep(1)
            yield gr.update(interactive=False), train_df, val_df, gr.update(value=f"Epochs: {current_epoch}/{epochs}, True Epochs: {current_true_epoch}/{true_epochs}"), example_img
        training = False
        del t
        gc.collect()
        t = threading.Thread(target=train)
        yield gr.update(interactive=True), train_df, val_df, gr.update(value=f"Epochs: {current_epoch}/{epochs}, True Epochs: {current_true_epoch}/{true_epochs}"), example_img

    def save_model(path):
        yield gr.update(value="Saving...")
        model.save(path)
        yield gr.update(value="Save Successful!")

    with gr.Blocks(title="Frame Generator") as demo:
        with gr.Accordion("Training"):
            with gr.Row():
                train_graph = gr.LinePlot(label="Training Loss", value=train_df, x="Epochs", y="Loss")
                val_graph = gr.LinePlot(label="Validation Loss", value=val_df, x="Epochs", y="Loss")
            epoch_counter = gr.Label(value="Epochs: ?/?")
            train_button = gr.Button("Train")
            example_img_element = gr.Image()
        train_button.click(fn=start_training, inputs=None, outputs=[train_button, train_graph, val_graph, epoch_counter, example_img_element])
        with gr.Accordion("Save"):
            save_path = gr.Textbox(label="Save Path", value=f"./models/frame_generator_{args.encoded_dim}.pt")
            save_button = gr.Button("Save")
            save_status = gr.Textbox(label="Save Status")
        save_button.click(fn=save_model, inputs=[save_path], outputs=[save_status])

    host = "127.0.0.1" if "--host" not in sys.argv else sys.argv[sys.argv.index("--host") + 1]
    port = 7860 if "--port" not in sys.argv else int(sys.argv[sys.argv.index("--port") + 1])
    share = "--share" in sys.argv
    demo.launch(server_name=host, server_port=port, share=share)