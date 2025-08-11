
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import random  
from pybalmorel import IncFile, Balmorel

# ignore warnings 
import warnings
warnings.filterwarnings("ignore")


class ScenarioGenerator(nn.Module):
    def __init__(self, X_tensor=None, cond_tensor=None, input_shape=(24, 72), cond_dim=3, latent_dim=32, device='cpu', lr=0.0005, seed=42):
        super(ScenarioGenerator, self).__init__()

        self.seq_len, self.input_dim = input_shape
        self.flat_dim = self.seq_len * self.input_dim
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.device = device
        self.total_loss_history = []
        self.pretrain_loss_history = []
        self.obj_values= []
        self.seed = seed
        self.reconstruction_loss_total = None
        self.new_scenarios = None
        self.new_scenarios_list = []
        self.X_original = None
        self.reconstruction_scenarios = None


        torch.manual_seed(self.seed)
        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        np.random.seed(self.seed)
        random.seed(self.seed)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.flat_dim + cond_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.flat_dim),
        )

        # Initialize the optimizer here to avoid redefining it in each function
        self.optimizer_ = torch.optim.Adam(self.parameters(), lr=lr)
        self.X_tensor = X_tensor
        self.cond_tensor = cond_tensor
        
    def forward(self, x, c):
        x = x.view(x.size(0), -1)  # Flatten
        x_cat = torch.cat([x, c], dim=1)
        z = self.encoder(x_cat)
        z_cat = torch.cat([z, c], dim=1)
        x_recon = self.decoder(z_cat)
        return x_recon.view(-1, self.seq_len, self.input_dim), z
    
    
    def cae_loss(self, recon_x, x):
        """
        Reconstruction loss function
        """
        return nn.SmoothL1Loss()(recon_x, x)
    
    def generate_kday_blocks(self, df, k_days=1):
        """
        Groups the input DataFrame into non-overlapping k-day sequences (each day = 24 hours).
        
        Parameters:
            df: pd.DataFrame — must contain 'WY', 'SSS', 'TTT' + features
            k_days: int — how many days per group (1 = 24h, 2 = 48h, etc.)
            condition_columns: list — columns to keep as condition (e.g., year, week)
        
        Returns:
            X_tensor: torch.Tensor — shape (N, k_days * 24, num_features)
            cond_tensor: torch.Tensor — optional condition tensor (N, len(condition_columns) + 1)
        """
        df = df.copy()
        df.fillna(0, inplace=True)
        #print("NaNs in df:", df.isna().any().any())
        # Extract numerical columns
        df['year'] = df['WY'].astype(int)
        df['week'] = df['SSS'].str.extract(r'S(\d+)').astype(int)
        df['hour'] = df['TTT'].str.extract(r'T(\d+)').astype(int)

        # Day index within the week: T001–T024 → day 1, etc.
        df['day_in_week'] = ((df['hour'] - 1) // 24) + 1

        # Compute rolling block id: every `k_days` days becomes 1 group
        df['block_in_week'] = ((df['day_in_week'] - 1) // k_days) + 1

        # Filter full-length blocks only
        group = df.groupby(['year', 'week', 'block_in_week'])
        df = group.filter(lambda x: len(x) == 24 * k_days)

        # Collect feature columns
        feature_cols = df.columns.difference(['WY', 'SSS', 'TTT', 'year', 'week', 'hour', 'day_in_week', 'block_in_week'])

        # get those columns that are not constant
        feature_cols = feature_cols[df[feature_cols].nunique() > 1]
        
       

        # Check if we have enough features
        if len(feature_cols) == 0:
            raise ValueError("No valid feature columns found. Ensure there are non-constant features.")

        print(f"Using {len(feature_cols)} features for {k_days}-day blocks.")

        # make an empty DataFrame with column names as feature_cols for later use
        self.empty_df = pd.DataFrame(columns=feature_cols)
        

        # Build sample blocks
        samples = []
        conditions = []
        for (year, week, block), group in df.groupby(['year', 'week', 'block_in_week']):
            #X_block = group.sort_values('hour')[feature_cols].values  # ensure hour order
            X_block = group[feature_cols].values
            samples.append(X_block)
            # Optional: add [year, week, block] as condition
            conditions.append([year, week, block])

        # Convert to tensors
        X = np.stack(samples)  # shape: (N_blocks, k*24, num_features)
        conds = np.array(conditions)  # shape: (N_blocks, 3)

        # X_tensor = torch.tensor(X, dtype=torch.float32)
        # cond_tensor = torch.tensor(conds, dtype=torch.float32)

        return X, conds
    
    def load_and_process_data(self, file_path, k_days=1):
        """
        Load and process data from a CSV file.
        
        Parameters:
            file_path: str — path to the CSV file
            k_days: int — number of days per block (1 = 24h, 2 = 48h, etc.)
        
        Returns:
            X_tensor: torch.Tensor — shape (N, k_days * 24, num_features)
            cond_tensor: torch.Tensor — optional condition tensor (N, len(condition_columns) + 1)
        """
        
        print("-------------------------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------------------------")
        print("Loading and processing data from file:", file_path)
        
        df = pd.read_csv(file_path)
        
    
        X, conds = self.generate_kday_blocks(df, k_days=k_days)
        self.X_original = X.copy()  # Store original data for later use
        
        X_log = np.log1p(X)
        self.scaler = StandardScaler()
        X_log_scaled = self.scaler.fit_transform(X_log.reshape(-1, X_log.shape[-1])).reshape(X_log.shape)

        self.X_tensor = torch.tensor(X_log_scaled, dtype=torch.float32)
        self.cond_tensor = torch.tensor(conds, dtype=torch.float32)
    
    
    def pretrain(self, epochs=20, batch_size=32):
        """
        Pretraining phase using the original reconstruction loss
        """
        if self.X_tensor is None or self.cond_tensor is None:
            raise ValueError("X_tensor and cond_tensor are not set. Please provide the data when initializing the model.")

        dataset = TensorDataset(self.X_tensor, self.cond_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
    
        self.train()
        print("-------------------------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------------------------")
        print("Pretraining the model...")
        print("**************************************************************************")
        print(f"Pretraining on {len(loader)} batches with batch size {batch_size} for {epochs} epochs.")

        for epoch in range(epochs):
            total_loss = 0
            for x_batch, c_batch in loader:
                x_batch = x_batch.to(self.device)
                c_batch = c_batch.to(self.device)
                self.optimizer_.zero_grad()
                recon_x, _ = self(x_batch, c_batch)
                loss = self.cae_loss(recon_x, x_batch)
                loss.backward()
                self.optimizer_.step()
                total_loss += loss.item()
            avg_loss = total_loss  # No need to divide by len(loader)
            self.pretrain_loss_history.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

        


    def generate_scenario(self, batch_size=32, n_scenarios=10):
        """
        Update phase where an external reward (obj_value) is used to calculate additional loss.
        K-means clustering is applied to the latent vectors and conditions before passing them through the decoder.
        """
        if self.X_tensor is None or self.cond_tensor is None:
            raise ValueError("X_tensor and cond_tensor are not set. Please provide the data when initializing the model.")

        dataset = TensorDataset(self.X_tensor, self.cond_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train()
        self.empty_df.drop(self.empty_df.index, inplace=True)
        # remove column 'scenario_id' and 'time_step' if they exist
        if 'scenario_id' in self.empty_df.columns:
            self.empty_df.drop(columns=['scenario_id', 'time_step'], inplace=True, errors='ignore')

        #print(self.empty_df)

        all_z_batches = []  # To store all latent vectors (z)
        all_cond_batches = []  # To store all condition vectors (cond_1day)
        all_recon_batches = []  # To store all reconstruction batches
        
        recon_loss_total = []  # Initialize total reconstruction loss

        for x_batch, c_batch in loader:
            x_batch = x_batch.to(self.device)
            c_batch = c_batch.to(self.device)
            self.optimizer_.zero_grad()
            
            # Calculate the original reconstruction loss
            recon_x, z = self(x_batch, c_batch)
            recon_loss = self.cae_loss(recon_x, x_batch)
            recon_loss_total.append(recon_loss)

            # Store all z-batches and c-batches for later K-means clustering
            all_z_batches.append(z.cpu().detach().numpy())
            all_cond_batches.append(c_batch.cpu().detach().numpy())
            all_recon_batches.append(recon_x.cpu().detach().numpy())
            
        # stack all reconstruction losses
        self.reconstruction_loss_total = torch.stack(recon_loss_total).mean()
            
        
        # After processing all batches, apply K-means to the latent space (all_z_batches) and conditions (all_cond_batches)
        all_z_batches = np.concatenate(all_z_batches, axis=0)  # Concatenate all z-batches
        all_cond_batches = np.concatenate(all_cond_batches, axis=0)  # Concatenate all condition batches
        all_recon_batches = np.concatenate(all_recon_batches, axis=0)  # Concatenate all reconstruction batches
        
        # reshape all_recon_batches to (N, seq_len, input_dim)
        all_recon_batches = all_recon_batches.reshape(-1, self.seq_len, self.input_dim)
        # transform to original scale
        all_recon_batches = np.expm1(self.scaler.inverse_transform(all_recon_batches.reshape(-1, self.input_dim)).reshape(all_recon_batches.shape))
        self.reconstruction_scenarios = all_recon_batches  # Store the reconstruction scenarios for later use

        # Apply K-means clustering on latent vectors (z)
        kmeans_z = KMeans(n_clusters=n_scenarios, random_state=self.seed)
        kmeans_z.fit(all_z_batches)

        # Apply K-means clustering on conditions (c_batch)
        kmeans_cond = KMeans(n_clusters=n_scenarios, random_state=self.seed)
        kmeans_cond.fit(all_cond_batches)

        # Use the centroids from K-means and pass them through the decoder to generate new scenarios
        centroids_z = torch.tensor(kmeans_z.cluster_centers_, dtype=torch.float32).to(self.device)
        centroids_cond = torch.tensor(kmeans_cond.cluster_centers_, dtype=torch.float32).to(self.device)

        # Concatenate the centroids of latent vectors and conditions
        centroids_combined = torch.cat([centroids_z, centroids_cond], dim=1)  # Concatenate z and cond centroids

        # Generate new scenarios from the combined centroids
        self.new_scenarios = self.decoder(centroids_combined)  # Get new samples from centroids
        self.new_scenarios = self.new_scenarios.view(-1, self.seq_len, self.input_dim).cpu().detach().numpy()

        self.new_scenarios = np.expm1(self.scaler.inverse_transform(self.new_scenarios.reshape(-1, self.new_scenarios.shape[-1])).reshape(self.new_scenarios.shape))
        self.new_scenarios_list.append(self.new_scenarios)
        
        
        new_rows = []
        # Loop through new_scenarios to create the new rows
        for i in range(self.new_scenarios.shape[0]):  # For each scenario
            for j in range(self.new_scenarios.shape[1]):  # For each time step
                new_row = pd.Series(self.new_scenarios[i, j, :], index=self.empty_df.columns)
                new_row['scenario_id'] = i+1  # Assign scenario_id based on the index of scenario
                new_row['time_step'] = j+1  # Assign time_step based on the index of time step
                new_rows.append(new_row)

        # Convert the list of rows into a DataFrame
        new_df = pd.DataFrame(new_rows)

        # Append the new DataFrame to the existing model.empty_df
        self.empty_df = pd.concat([self.empty_df, new_df], ignore_index=True)

        # Ensure the scenario_id and time_step columns are integers
        self.empty_df['scenario_id'] = self.empty_df['scenario_id'].astype(int)
        self.empty_df['time_step'] = self.empty_df['time_step'].astype(int)

        # make sure cenario_id and time_step are first two columns
        cols = ['scenario_id', 'time_step'] + [col for col in self.empty_df.columns if col not in ['scenario_id', 'time_step']]
        self.empty_df = self.empty_df[cols]
        
        return self.new_scenarios, self.empty_df

    def update(self, obj_value, epoch=None):

        if epoch == 0:
            print("-------------------------------------------------------------------------------------------------------------------------")
            print("-------------------------------------------------------------------------------------------------------------------------")
            print("Updating model with the feedback...")
            print("**************************************************************************")


        self.obj_values.append(obj_value)
        
        policy_loss = obj_value  # Policy loss based on the reward (negative value)
        
        total_batch_loss = self.reconstruction_loss_total + policy_loss
        
        self.optimizer_.zero_grad()
        # Backpropagate the total loss
        total_batch_loss.backward()
        self.optimizer_.step()
        self.total_loss_history.append(total_batch_loss.item())
        print(f'Epoch {epoch+1} - ' if epoch is not None else '', f"Total Loss: {total_batch_loss.item():.6f} | Reconstruction Loss: {self.reconstruction_loss_total.item():.6f} | Policy Loss: {policy_loss.item():.6f}")

    
    def plot_loss_history(self):
        """
        Plot the pretraining loss history and the total loss history in separate plots.
        """

        plt.plot(self.pretrain_loss_history, label='Pretrain Loss')
        plt.title("Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        plt.plot(self.total_loss_history, label='Total Loss')
        plt.plot(self.obj_values, label='Objective Values', linestyle='--')
        plt.title("Total Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    
    def plot_scenarios(self, selecting_feature_index=0):
        """
        Plot the generated scenarios vs the original data.
        
        Parameters:
            new_scenarios: np.ndarray — shape (N, seq_len, input_dim)
        """
        
        if self.new_scenarios is None:
            raise ValueError("No new scenarios generated. Please call generate_scenario() first.")
        
        plt.figure(figsize=(12, 6))
        for i in range(self.X_original.shape[0]):
            plt.plot(self.X_original[i, :, selecting_feature_index], alpha=0.1)

        for i in range(self.new_scenarios.shape[0]):
            plt.plot(self.new_scenarios[i, :, selecting_feature_index], alpha=0.7)

        plt.title("Generated Scenario vs Original Scenario")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        
    def save_model(self, file_path):
        """
        Save the model's state_dict, optimizer's state_dict, and custom attributes to the specified file path.
        """
        torch.save({
            'model_state_dict': self.state_dict(),  # Save model parameters
            'optimizer_state_dict': self.optimizer_.state_dict(),  # Save optimizer parameters
            'total_loss_history': self.total_loss_history,  # Save custom attributes
            'pretrain_loss_history': self.pretrain_loss_history,  # Pretraining loss history
            'obj_values': self.obj_values,  # Objective values
            'new_scenarios_list': self.new_scenarios_list,  # Generated scenarios
            'X_original': self.X_original,  # Original X tensor
            'cond_tensor': self.cond_tensor,  # Condition tensor
            'X_tensor': self.X_tensor,  # Original X tensor
        }, file_path)
        
        print("-------------------------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------------------------")
        print(f'Model and additional attributes saved to {file_path}')
        
        
    def load_model(self, file_path):
        """
        Load the model's state_dict and optimizer's state_dict from the specified file path.
        """
        checkpoint = torch.load(file_path, weights_only=False)
        
        # Load model state
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer_.load_state_dict(checkpoint['optimizer_state_dict'])
        
        
        # Load custom attributes
        self.total_loss_history = checkpoint['total_loss_history']  # Load custom attributes
        self.pretrain_loss_history = checkpoint['pretrain_loss_history']
        self.obj_values = checkpoint['obj_values']
        self.new_scenarios_list = checkpoint['new_scenarios_list']
        self.X_original = checkpoint['X_original']
        self.cond_tensor = checkpoint['cond_tensor']
        self.X_tensor = checkpoint['X_tensor']
        
        print(f'Model and additional attributes loaded from {file_path}')
        
        
    def plot_reconstruction_scenarios(self, features_to_plot, sample_index=0):
        """
        Plot the reconstruction scenarios vs the original data.
        """
        
        #calculat e len of columns and subplots
        num_features = len(features_to_plot)
        num_cols = 3
        num_rows = (num_features + num_cols - 1) // num_cols  #
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        axes = axes.flatten()  # Flatten the axes array for easy indexing
        for i, feature in enumerate(features_to_plot):
            if i >= len(axes):
                break
            axes[i].plot(self.X_original[sample_index, :, feature], label='Original', alpha=0.5)
            axes[i].plot(self.reconstruction_scenarios[sample_index, :, feature], label='Reconstructed', alpha=0.7)
            
            axes[i].set_title(f'Feature {feature}')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True)  
        
        # show the lplots
        plt.tight_layout()
        plt.show()
        print("Reconstruction scenarios plotted for sample index:", sample_index)
        print("Features plotted:", features_to_plot)
        print("Original data shape:", self.X_original.shape)
        print("Reconstruction scenarios shape:", self.reconstruction_scenarios.shape)
        print("Sample index:", sample_index)
        print("Number of features:", num_features)
        
def convert_to_incfiles(new_scenarios_df: pd.DataFrame,
                        scenario: str,
                        balmorel_model_path: str = 'Balmorel',
                        gams_system_directory: str | None = None):
    
    
    # Get parameters and temporal resolution
    parameters = new_scenarios_df.columns[2:] # pop scenario id and time_step
    balmorel_time_index = [f"S{row['scenario_id']:02d} . T{row['time_step']:03d}" for i,row  in new_scenarios_df.loc[:, ['scenario_id', 'time_step']].iterrows()]
    
    # Load Balmorel input data for descriptions and set names
    model = Balmorel(balmorel_model_path, gams_system_directory=gams_system_directory)
    model.load_incfiles(scenario)
    
    placeholder_parameter = ''
    for parameter in parameters:

        # Get elements and parameter_name
        elements = parameter.split('|')
        parameter_name = elements.pop(0)
        
        # Skip if we already processed this parameter
        if placeholder_parameter == parameter_name:
            continue
        else:
            placeholder_parameter = parameter_name
        
        # Find all values of this parameter
        idx=new_scenarios_df.columns.str.find(parameter_name) == 0
        
        # Get sets, values and explanation
        sets = model.input_data[scenario][parameter_name].domains_as_strings
        table_or_param = 'PARAMETER' if len(sets) == 1 else 'TABLE'
        text = model.input_data[scenario][parameter_name].text
        df = new_scenarios_df.loc[:, idx]
        df.index = balmorel_time_index   
        df.columns = df.columns.str.replace(parameter_name+'|', '').str.replace('|', ' . ')
        
        IncFile(
            name=parameter_name,
            prefix=f"""{table_or_param} {parameter_name}({','.join(sets)}) "{text}" \n""",
            body=df,
            suffix='\n;',
            path=balmorel_model_path + '/operun/data'
        ).save()