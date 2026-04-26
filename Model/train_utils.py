import torch
from wandb import Run

def train(model, loader, optimizer, epochs, run:Run):

    print("=== Stage 1: Training backbone ===")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_acc = 0.0

        correct_conf = 0.0
        incorrect_conf = 0.0
        all_conf = 0.0

        num_batches = len(loader)
        for batch in loader:
            optimizer.zero_grad()
            out  = model(batch.x.unsqueeze(1), batch.edge_index, batch.batch)
            loss, acc, conf = model.loss(out, batch.y.squeeze())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc
            correct_conf += conf['correct']
            incorrect_conf += conf['incorrect']
            all_conf += conf['all']


        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
            
        print(f"  Epoch {epoch+1}  loss={avg_loss.item():.4f}")
        run.log({"acc": avg_acc, "loss": avg_loss})