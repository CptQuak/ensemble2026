import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Importy z naszego potoku Fazy 2
from src.segmentation.unet_resnet34 import ResNet34UNet
from src.segmentation.dataset import ECGDataset
from src.segmentation.loss import DiceBCELoss

# obrazy trenowane trzeba obrócić!!!!
# i tak dalej i tak dalej

def train_model():
    """
    Skrypt treningowy dla modelu ResNet34 U-Net.
    Faza 2: Semantic Segmentation z uodpornieniem na Overlapping Leads.
    """
    # 1. Konfiguracja hiperparametrów
    data_dir = "data/train"
    epochs = 10
    batch_size = 2
    learning_rate = 1e-4
    target_size = (512, 512)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"[{device.type.upper()}] Inicjalizacja treningu na urządzeniu...")

    # 2. Przygotowanie danych z reżimem augmentacji z dokumentu (Adversarial Overlap)
    train_dataset = ECGDataset(data_dir=data_dir, is_train=True, target_size=target_size)
    
    if len(train_dataset) == 0:
        print(f"Błąd: Brak obrazów w folderze '{data_dir}'. Trening przerwany.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    print(f"Znaleziono próbki treningowe: {len(train_dataset)}")

    # 3. Architektura Sieci, Optymalizator i Funkcja Straty
    model = ResNet34UNet(in_channels=3, out_channels=1, pretrained=True).to(device)
    
    # Funkcja straty zdefiniowana przez Karbasiego: L = L_Dice + 0.5*L_BCE
    criterion = DiceBCELoss(bce_weight=0.5)
    
    # Optymalizator
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Rejestrator metryk
    history = {'loss': []}

    # 4. Pętla Uczenia
    os.makedirs("models", exist_ok=True)
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # tqdm dla wizualizacji pętli
        progress_bar = tqdm(train_loader, desc=f"Epoka {epoch+1}/{epochs}")
        
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Obliczanie straty (strata Dice ignoruje fałszywe pogrubienia)
            loss = criterion(outputs, masks)
            
            # Backward pass & optymalizacja
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({'strata': f"{loss.item():.4f}"})
            
        epoch_loss /= len(train_loader.dataset)
        history['loss'].append(epoch_loss)
        
        print(f"Epoka {epoch+1} Zakończona. Średnia strata (Dice-BCE): {epoch_loss:.4f}")
        
        # 5. Zapisywanie najlepszych wag
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "models/unet_resnet34_best.pth")
            print(" -> [Zapisano nowe, najlepsze wagi]")

    # 6. Diagnostyka - wykres krzywej uczenia
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), history['loss'], marker='o', color='red', label='Trening (Dice-BCE)')
    plt.title("Krzywa Uczenia - U-Net ResNet34")
    plt.xlabel("Epoka")
    plt.ylabel("Wartość Straty")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("models/training_history.png")
    plt.close()
    
    print("\n[SUKCES] Trening zakończony. Wagi znajdują się w folderze 'models/'.")

if __name__ == '__main__':
    train_model()