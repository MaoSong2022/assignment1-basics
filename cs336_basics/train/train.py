import os
import time
import wandb
import argparse
import json
import numpy as np
from torch.utils.data import DataLoader
import torch

import cs336_basics.model.model as transformer_model
from cs336_basics.optimizer import optimizer as MyOptimizer
from cs336_basics.train import serialization
from cs336_basics.model import utils
from cs336_basics.dataset import data


def get_args():
    parser = argparse.ArgumentParser(
        description="Transformer Training Configuration", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Config File Support ---
    parser.add_argument("--config", type=str, help="Path to a JSON config file to load args from")

    # --- Project Metadata ---
    group_gen = parser.add_argument_group("Metadata")
    group_gen.add_argument("--project_name", type=str, default="cs336_assignment1")
    group_gen.add_argument("--run_name", type=str, default="tiny_story_run_5e-1")

    # --- Data Paths ---
    group_data = parser.add_argument_group("Data")
    group_data.add_argument("--train_file_path", type=str, required=False)
    group_data.add_argument("--valid_file_path", type=str, required=False)

    # --- Model Architecture ---
    group_arch = parser.add_argument_group("Architecture")
    group_arch.add_argument("--vocab_size", type=int, default=10000)
    group_arch.add_argument("--num_layers", type=int, default=4)
    group_arch.add_argument("--d_model", type=int, default=512)
    group_arch.add_argument("--num_heads", type=int, default=16)
    group_arch.add_argument("--d_ff", type=int, default=1344)
    group_arch.add_argument("--max_seq_len", type=int, default=256)
    group_arch.add_argument("--theta", type=float, default=10000, help="Base for RoPE")

    # --- Training & Optimization ---
    group_train = parser.add_argument_group("Training")
    group_train.add_argument("--epochs", type=int, default=1)
    group_train.add_argument("--batch_size", type=int, default=512)
    group_train.add_argument("--num_workers", type=int, default=4)
    group_train.add_argument("--eval_interval_steps", type=int, default=500)
    group_train.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.95])
    group_train.add_argument("--adamw_eps", type=float, default=1e-8)
    group_train.add_argument("--weight_decay", type=float, default=0.1)
    group_train.add_argument("--max_learning_rate", type=float, default=5e-1)
    group_train.add_argument("--min_learning_rate", type=float, default=5e-5)
    group_train.add_argument("--warmup_iters", type=int, default=500)
    group_train.add_argument("--cosine_cycle_iters", type=int, default=4000)
    group_train.add_argument("--max_l2_norm", type=float, default=1.0)
    group_train.add_argument("--clipping_eps", type=float, default=1e-6)
    group_train.add_argument("--training_steps", type=int, default=None, help="total training steps")

    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config_dict = json.load(f)
            # Override defaults with JSON values
            for key, value in config_dict.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                else:
                    print(f"Warning: Unknown key in config file: {key}")

    return args


def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: Training on CPU. This will be extremely slow.")

    wandb.init(
        project=args.project_name,
        name=args.run_name,
        config=vars(args),
    )

    # initialize model
    model = transformer_model.Model(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        theta=args.theta,
    )
    model.to(device)

    # initialize data set
    train_dataset = data.MemmapDataset(
        file_path=args.train_file_path,
        seq_len=args.max_seq_len,
        dtype=np.uint16,
    )
    validation_dataset = data.MemmapDataset(
        file_path=args.valid_file_path,
        seq_len=args.max_seq_len,
        dtype=np.uint16,
    )

    # initialize training stuff
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = MyOptimizer.AdamW(
        model.parameters(),
        lr=args.max_learning_rate,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
        eps=args.adamw_eps,
    )

    steps_per_epoch = len(train_loader)
    total_training_steps = args.epochs * steps_per_epoch if args.training_steps is None else args.training_steps
    print({"total_training_steps": total_training_steps})

    best_val_loss = float("inf")
    global_step = 0
    start_train_time = time.time()

    vocab_size =args.vocab_size

    for epoch in range(args.epochs):
        epoch_start_time = time.time()  # record time used of current epoch
        model.train()

        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            batch_start_time = time.time()
            inputs, targets = inputs.to(device), targets.to(device)

            # update learning rate
            current_lr = MyOptimizer.get_lr_cosine_schedule(
                global_step,
                max_learning_rate=args.max_learning_rate,
                min_learning_rate=args.min_learning_rate,
                warmup_iters=args.warmup_iters,
                cosine_cycle_iters=args.cosine_cycle_iters,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            # forward pass
            logits = model(inputs)
            loss = utils.cross_entropy_loss(logits.view(-1, vocab_size), targets.view(-1))

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            utils.gradient_clipping(
                model.parameters(),
                max_l2_norm=args.max_l2_norm,
                eps=args.clipping_eps,
            )
            optimizer.step()

            batch_time = time.time() - batch_start_time
            wandb.log(
                {
                    "train/batch_loss": loss.item(),  # 批次损失
                    "train/current_lr": current_lr,  # 当前学习率
                    "train/batch_time": batch_time,  # 批次耗时
                    "train/global_step": global_step,  # 全局步数
                    "train/epoch": epoch + 1,  # 当前epoch
                    "train/tokens_per_second": inputs.size(0) * args.max_seq_len / batch_time,
                },
                step=global_step,
            )  # step指定为全局步数，保证可视化横轴对齐

            if batch_idx % 100 == 0:
                avg_loss = train_loss / (batch_idx + 1)
                print(f"Train Batch {batch_idx} | LR: {current_lr:.6f} | Avg Loss: {avg_loss:.4f}")

            if global_step > 0 and global_step % args.eval_interval_steps == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(val_loader):
                        inputs, targets = inputs.to(device), targets.to(device)
                        # forward pass
                        logits = model(inputs)

                        loss = utils.cross_entropy_loss(logits.view(-1, vocab_size), targets.view(-1))

                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)

                wandb.log(
                    {
                        "val/interval_avg_loss": avg_val_loss,  # Epoch平均验证损失
                        "train/best_val_loss": best_val_loss,  # 最优验证损失
                    },
                    step=global_step,
                )

                # save the checkpoint
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_path = f"checkpoints/{args.run_name}/best_llm_model.pt"
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    serialization.save_checkpoint(model, optimizer, epoch, save_path)
                    print(f"best checkpoint has been saved to: {save_path}")

                    wandb.log(
                        {
                            "model/best_val_loss": best_val_loss,
                            "model/best_steps": epoch,
                            "model/save_path": save_path,
                        },
                        step=global_step,
                    )

            # support customized training iterations
            if args.training_steps is not None and args.training_steps > 0 and global_step > args.training_steps:
                break

        # support customized training iterations
        if args.training_steps is not None and args.training_steps > 0 and global_step > args.training_steps:
            break

        avg_train_loss = train_loss / len(train_loader)
        epoch_train_time = time.time() - epoch_start_time
        wandb.log(
            {
                "train/epoch_avg_loss": avg_train_loss,
                "train/epoch_time": epoch_train_time,
            }
        )

    total_train_time = time.time() - start_train_time
    wandb.log(
        {
            "train/total_train_time": total_train_time,
            "train/total_train_time_hours": total_train_time / 3600,
        }
    )

    wandb.finish()
    print(f"\nTraining is complete, total training time: {total_train_time / 3600:.2f}")
    return model


if __name__ == "__main__":
    main()
