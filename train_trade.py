import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from ai_games.learn_settlers.ai.team5.nn_bot.models.model_trade import TradeNet
import numpy as np

base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "..", "..", "datasets", "trade_dataset.pt")

data = torch.load(data_path)
x_data = torch.stack([x for x, _ in data])
y_data = torch.stack([y for _, y in data])
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=64)

batch_losses = []
dataset = TensorDataset(x_data, y_data)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = TradeNet(input_size=x_data.shape[1])
criterion = nn.MSELoss()  # Für Regression auf Angebot/Nachfrage/Partner
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_losses = []
test_losses = []
for epoch in range(50):
    model.train()
    total_loss = 0
    for batch_x, batch_y in loader:
        offer_pred, request_pred, partner_pred = model(batch_x)
        offer_true = batch_y[:, :5]
        request_true = batch_y[:, 5:10]
        partner_true = batch_y[:, 10].long()  # Partner als Integer

        loss_offer = nn.MSELoss()(offer_pred, offer_true)
        loss_request = nn.MSELoss()(request_pred, request_true)
        loss_partner = nn.CrossEntropyLoss()(partner_pred, partner_true)
        loss = loss_offer + loss_request + 0.5 * loss_partner

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_losses.append(loss.item()) 
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
        
    #model.load_state_dict(torch.load(os.path.join(base_dir, "..", "..", "outputs", "trade_model.pt")))
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            offer_pred, request_pred, partner_pred = model(batch_x)
            offer_true = batch_y[:, :5]
            request_true = batch_y[:, 5:10]
            partner_true = batch_y[:, 10].long()
            loss_offer = nn.MSELoss()(offer_pred, offer_true)
            loss_request = nn.MSELoss()(request_pred, request_true)
            loss_partner = nn.CrossEntropyLoss()(partner_pred, partner_true)
            loss = loss_offer + loss_request + 0.5 * loss_partner
            total_test_loss += loss.item()
    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss) 
    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Test Loss={avg_test_loss:.4f}")        
    torch.save(model.state_dict(), os.path.join(base_dir, "..", "..", "outputs", "trade_model.pt"))
    with torch.no_grad():
        correct_partner = 0
        total = 0
        for batch_x, batch_y in test_loader:
            _, _, partner_pred = model(batch_x)
            partner_true = batch_y[:, 10].long()
            correct_partner += (torch.argmax(partner_pred, dim=1) == partner_true).sum().item()
            total += batch_x.size(0)
    print(f"Partner-Accuracy: {correct_partner / total:.2%}")
    with torch.no_grad():
        for i in range(3):
            idx = np.random.randint(0, len(X_test))
            x = X_test[idx].unsqueeze(0)
            y_true = Y_test[idx]
            offer_pred, request_pred, partner_pred = model(x)
            print("Vorhersage Angebot:", torch.round(offer_pred))
            print("Vorhersage Nachfrage:", torch.round(request_pred))
            print("Vorhersage Partner:", torch.argmax(partner_pred).item())
            print("Echtes Angebot:", y_true[:5])
            print("Echte Nachfrage:", y_true[5:10])
            print("Echter Partner:", int(y_true[10].item()))
            print("-" * 30)   
# Plot für Epochen
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Trainings- und Test-Loss Trade-Bot")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "..", "..", "outputs", "trade_loss_plot.png"))
plt.show()

# Plot für Batches
plt.figure()
plt.plot(batch_losses, label="Batch Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Batch Loss Verlauf")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "..", "..", "outputs", "trade_batch_loss_plot.png"))
plt.show() 
def moving_average(x, w=50):
    return np.convolve(x, np.ones(w), mode='valid') / w
plt.plot(moving_average(batch_losses), label="Batch Loss (MA)")