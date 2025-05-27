import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE # For visualizing embeddings
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- Configuration & Constants ---
CLEANED_DATA_FILE = 'cleaned_sales_data.pkl'
MODEL_SAVE_PATH = 'sales_predictor_model.pth' # PyTorch model extension
SCALER_NUM_FEATURES_SAVE_PATH = 'scaler_numerical_features.pkl'
DISH_TO_IDX_SAVE_PATH = 'dish_to_idx.pkl'
IDX_TO_DISH_SAVE_PATH = 'idx_to_dish.pkl'
DISH_AVG_SALES_SAVE_PATH = 'dish_avg_sales.pkl'

UNKNOWN_DISH_TOKEN = 'UNK_DISH'
EMBEDDING_DIM = 16
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 2000
PATIENCE_EARLY_STOPPING = 20

# --- 0. Device Configuration (MPS for Mac, CUDA, or CPU) ---
print("--- 0. Device Configuration ---")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal Performance Shaders (MPS) backend.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA backend.")
else:
    device = torch.device("cpu")
    print("Using CPU backend.")


# --- 1. Load Cleaned Data ---
print("\n--- 1. Loading Cleaned Data ---")
try:
    df = pd.read_pickle(CLEANED_DATA_FILE)
    print(f"Successfully loaded '{CLEANED_DATA_FILE}'. Shape: {df.shape}")
except FileNotFoundError:
    print(f"ERROR: Cleaned data file '{CLEANED_DATA_FILE}' not found. Please run the cleanup script first.")
    exit()

# --- 2. Feature Engineering (Identical to TensorFlow version) ---
print("\n--- 2. Feature Engineering ---")
# 2.1 Dish Name Mapping & Vocabulary
all_dishes_p_cols = pd.concat([
    df['ProductName_1'], df['ProductName_2'],
    df['ProductName_3'], df['ProductName_4']
]).dropna().unique().tolist()
if UNKNOWN_DISH_TOKEN not in all_dishes_p_cols:
    all_dishes_list = [UNKNOWN_DISH_TOKEN] + all_dishes_p_cols
else:
    all_dishes_list = all_dishes_p_cols
dish_to_idx = {dish: i for i, dish in enumerate(all_dishes_list)}
idx_to_dish = {i: dish for dish, i in dish_to_idx.items()}
vocab_size = len(dish_to_idx)
print(f"Vocabulary size (including UNK): {vocab_size}")

# 2.2 Individual Dish "Inherent Popularity" Features
sales_data = []
for i in range(1, 5):
    prod_col, count_col = f'ProductName_{i}', f'count_{i}'
    if prod_col in df.columns and count_col in df.columns:
        subset = df[[prod_col, count_col]].copy()
        subset.rename(columns={prod_col: 'DishName', count_col: 'Count'}, inplace=True)
        sales_data.append(subset)
if not sales_data: exit("ERROR: No product/count columns found for popularity calculation.")
all_sales_df = pd.concat(sales_data).dropna()
dish_avg_sales = all_sales_df.groupby('DishName')['Count'].mean().to_dict()
fallback_avg_sale = np.mean(list(dish_avg_sales.values())) if dish_avg_sales else 0
if UNKNOWN_DISH_TOKEN not in dish_avg_sales:
    dish_avg_sales[UNKNOWN_DISH_TOKEN] = fallback_avg_sale
df['AvgSales_P1'] = df['ProductName_1'].map(dish_avg_sales).fillna(dish_avg_sales.get(UNKNOWN_DISH_TOKEN, fallback_avg_sale))
df['AvgSales_P2'] = df['ProductName_2'].map(dish_avg_sales).fillna(dish_avg_sales.get(UNKNOWN_DISH_TOKEN, fallback_avg_sale))
df['PopularityDiff_P1P2'] = (df['AvgSales_P1'] - df['AvgSales_P2']).abs()
df['PopularityRatio_P1P2'] = df['AvgSales_P1'] / (df['AvgSales_P2'] + 1e-6)

# 2.3 DayOfWeek Encoding
df = pd.get_dummies(df, columns=['DayOfWeek'], prefix='Day', dtype=float) # float for PyTorch tensors

# 2.4 Total Sales Feature
df['Expected_Total_Count'] = df['total_count']

# 2.5 Special Rule for "Japanese Katsu Curry"
friday_col_name = 'Day_Friday'
if friday_col_name not in df.columns:
    print(f"WARNING: Column '{friday_col_name}' not found. Katsu rule might not work.")
    df['Is_Katsu_Friday'] = 0.0 # float
else:
    df['Is_Katsu_Friday'] = (((df['ProductName_1'] == 'Japanese Katsu Curry') | \
                             (df['ProductName_2'] == 'Japanese Katsu Curry')) & \
                             (df[friday_col_name] == 1.0)).astype(float)

# 2.6 Prepare Indices and Numerical Features
df['P1_idx'] = df['ProductName_1'].map(dish_to_idx).fillna(dish_to_idx[UNKNOWN_DISH_TOKEN]).astype(int)
df['P2_idx'] = df['ProductName_2'].map(dish_to_idx).fillna(dish_to_idx[UNKNOWN_DISH_TOKEN]).astype(int)
day_cols = [col for col in df.columns if col.startswith('Day_')]
numerical_feature_cols = ['AvgSales_P1', 'AvgSales_P2', 'PopularityDiff_P1P2',
                            'PopularityRatio_P1P2', 'Expected_Total_Count', 'Is_Katsu_Friday'] + day_cols
X_numerical = df[numerical_feature_cols].values.astype(np.float32) # Ensure float32 for PyTorch
y_target = df[['count_1', 'count_2']].values.astype(np.float32) # Ensure float32

# --- 3. Data Scaling and Splitting ---
print("\n--- 3. Data Scaling and Splitting ---")
scaler_num_features = StandardScaler()
X_numerical_scaled = scaler_num_features.fit_transform(X_numerical)

# Split data
indices_P1_train, indices_P1_val, \
indices_P2_train, indices_P2_val, \
num_features_train, num_features_val, \
y_train, y_val = train_test_split(
    df['P1_idx'].values, df['P2_idx'].values, X_numerical_scaled, y_target,
    test_size=0.2, random_state=42, shuffle=True
)

# Convert to PyTorch Tensors
# Indices should be LongTensors for embedding layers
tensor_p1_train = torch.LongTensor(indices_P1_train).to(device)
tensor_p2_train = torch.LongTensor(indices_P2_train).to(device)
tensor_num_features_train = torch.FloatTensor(num_features_train).to(device)
tensor_y_train = torch.FloatTensor(y_train).to(device)

tensor_p1_val = torch.LongTensor(indices_P1_val).to(device)
tensor_p2_val = torch.LongTensor(indices_P2_val).to(device)
tensor_num_features_val = torch.FloatTensor(num_features_val).to(device)
tensor_y_val = torch.FloatTensor(y_val).to(device)

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(tensor_p1_train, tensor_p2_train, tensor_num_features_train, tensor_y_train)
val_dataset = TensorDataset(tensor_p1_val, tensor_p2_val, tensor_num_features_val, tensor_y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

# --- 4. PyTorch Neural Network Model Design ---
print("\n--- 4. PyTorch Neural Network Model Design ---")
class SalesPredictorNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_numerical_features, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.4):
        super(SalesPredictorNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Calculate input dimension for the first dense layer
        # pair_embedding_dim + num_numerical_features
        self.fc1_input_dim = embedding_dim + num_numerical_features # if using Add/Multiply for embeddings
                                                                   # or 2 * embedding_dim if concatenating embeddings
        
        self.fc1 = nn.Linear(self.fc1_input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.output_layer = nn.Linear(hidden_dim2, 2) # 2 outputs for count_1 and count_2
        self.relu = nn.ReLU()

    def forward(self, idx_p1, idx_p2, numerical_features):
        emb_p1 = self.embedding(idx_p1).squeeze(1) # Remove the (batch, 1, dim) -> (batch, dim)
        emb_p2 = self.embedding(idx_p2).squeeze(1)

        # Order-agnostic combination
        pair_embedding = emb_p1 + emb_p2 # Element-wise addition
        # pair_embedding = emb_p1 * emb_p2 # Element-wise multiplication
        
        x = torch.cat((pair_embedding, numerical_features), dim=1)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        out = self.output_layer(x)
        out = self.relu(out) # Ensure non-negative counts
        return out

model = SalesPredictorNet(vocab_size, EMBEDDING_DIM, X_numerical_scaled.shape[1]).to(device)
print(model)

# Loss and Optimizer
criterion = nn.HuberLoss(delta=1.0) # PyTorch's HuberLoss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. Train the Model ---
print("\n--- 5. Train the Model ---")
train_losses, val_losses = [], []
train_maes, val_maes = [], [] # MAE per output (average)

best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train() # Set model to training mode
    running_loss = 0.0
    running_mae = 0.0
    
    for batch_idx, (p1_batch, p2_batch, num_feat_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(p1_batch, p2_batch, num_feat_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * p1_batch.size(0) # Weighted by batch size
        running_mae += torch.abs(outputs - y_batch).mean().item() * p1_batch.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_mae = running_mae / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    train_maes.append(epoch_mae)

    # Validation
    model.eval() # Set model to evaluation mode
    val_running_loss = 0.0
    val_running_mae = 0.0
    with torch.no_grad():
        for p1_batch, p2_batch, num_feat_batch, y_batch in val_loader:
            outputs = model(p1_batch, p2_batch, num_feat_batch)
            loss = criterion(outputs, y_batch)
            val_running_loss += loss.item() * p1_batch.size(0)
            val_running_mae += torch.abs(outputs - y_batch).mean().item() * p1_batch.size(0)
            
    epoch_val_loss = val_running_loss / len(val_loader.dataset)
    epoch_val_mae = val_running_mae / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    val_maes.append(epoch_val_mae)
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_loss:.4f}, Train MAE: {epoch_mae:.4f} | Val Loss: {epoch_val_loss:.4f}, Val MAE: {epoch_val_mae:.4f}")

    # Early Stopping
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        # Save the best model
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"   Best model saved with val_loss: {best_val_loss:.4f}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE_EARLY_STOPPING:
            print(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
            break

# Load best model weights for final evaluation / use
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.to(device) # Ensure it's on the correct device after loading

# --- 6. Basic Evaluation & Plotting History ---
print("\n--- 6. Basic Evaluation & Plotting History ---")
# Final evaluation on validation set (with best model loaded)
model.eval()
final_val_loss = 0
final_val_mae = 0
with torch.no_grad():
    for p1_batch, p2_batch, num_feat_batch, y_batch in val_loader:
        outputs = model(p1_batch, p2_batch, num_feat_batch)
        loss = criterion(outputs, y_batch)
        final_val_loss += loss.item() * p1_batch.size(0)
        final_val_mae += torch.abs(outputs - y_batch).mean().item() * p1_batch.size(0)

final_val_loss /= len(val_loader.dataset)
final_val_mae /= len(val_loader.dataset)
print(f"Final Validation Loss (Huber) with best model: {final_val_loss:.4f}")
print(f"Final Validation MAE (per output) with best model: {final_val_mae:.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss (Huber)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_maes, label='Training MAE')
plt.plot(val_maes, label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.savefig('training_history_pytorch.png')
print("Training history plot saved as 'training_history_pytorch.png'")

# --- 7. Visualize Embeddings ---
print("\n--- 7. Visualize Embeddings ---")
try:
    # Get embeddings from the trained model (ensure model is on CPU for this if needed)
    embedding_weights = model.embedding.weight.cpu().detach().numpy()
    
    num_dishes_to_plot = min(vocab_size, 100)
    indices_to_plot = np.random.choice(range(vocab_size), size=num_dishes_to_plot, replace=False)
    embeddings_subset = embedding_weights[indices_to_plot]
    labels_subset = [idx_to_dish[i] for i in indices_to_plot]

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels_subset)-2), n_iter=1000, learning_rate='auto', init='pca')
    embeddings_2d = tsne.fit_transform(embeddings_subset)

    plt.figure(figsize=(15, 12))
    for i, label in enumerate(labels_subset):
        x, y = embeddings_2d[i, :]
        plt.scatter(x, y)
        plt.annotate(label, (x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', fontsize=8)
    plt.title(f't-SNE visualization of Dish Embeddings (PyTorch - Subset of {num_dishes_to_plot})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.savefig('dish_embeddings_tsne_pytorch.png')
    print("t-SNE embedding visualization saved as 'dish_embeddings_tsne_pytorch.png'")
except Exception as e:
    print(f"Could not visualize embeddings: {e}")

# --- 8. Save Supporting Files (Scalers, Mappings) ---
print("\n--- 8. Save Supporting Files ---")
try:
    with open(SCALER_NUM_FEATURES_SAVE_PATH, 'wb') as f: pickle.dump(scaler_num_features, f)
    print(f"Numerical features scaler saved to {SCALER_NUM_FEATURES_SAVE_PATH}")
    with open(DISH_TO_IDX_SAVE_PATH, 'wb') as f: pickle.dump(dish_to_idx, f)
    print(f"dish_to_idx mapping saved to {DISH_TO_IDX_SAVE_PATH}")
    with open(IDX_TO_DISH_SAVE_PATH, 'wb') as f: pickle.dump(idx_to_dish, f)
    print(f"idx_to_dish mapping saved to {IDX_TO_DISH_SAVE_PATH}")
    with open(DISH_AVG_SALES_SAVE_PATH, 'wb') as f: pickle.dump(dish_avg_sales, f)
    print(f"dish_avg_sales mapping saved to {DISH_AVG_SALES_SAVE_PATH}")
except Exception as e:
    print(f"ERROR during saving supporting files: {e}")

print("\n--- Script Finished ---")