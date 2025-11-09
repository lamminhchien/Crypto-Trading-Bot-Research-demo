# AI-Powered Quantitative Trading Research Framework

This repository serves as a research framework for "Tom-AI," a project focused on applying advanced AI models for quantitative trading in Crypto and Forex markets.

The primary goal is not just to build a single bot, but to create a robust pipeline for **backtesting**, **evaluating**, and **benchmarking** different AI-driven trading strategies. This project is heavily focused on the "Research" aspect of quantitative analysis.

## Core Technologies & Concepts

* **AI Framework:** PyTorch
* **Core Architecture:** **Transformer (TFM)** [user_prompt] & Temporal Convolutional Networks (TCN) for feature extraction from market data.
* **Strategy Model:** Reinforcement Learning (Soft Actor-Critic) to train an agent that makes trading decisions.
* **MLOps Pipeline:** **Optuna** for Hyperparameter Optimization (HPO) and **Ray** for distributed, multi-GPU training.
* **Experimental Research:** Mamba (SSM) integration for benchmarking against Transformer-based models.
* **Data Science Stack:** Python, Pandas, NumPy, Scikit-learn.

## Methodology

This project follows a structured **quantitative research** workflow:

### 1. Data Collection & Curation
Collected and synthesized **datasets** from various sources (Kaggle). This involved extensive cleaning, filtering [user_prompt], and feature engineering to create high-quality, multi-timeframe datasets suitable for training deep learning models.

### 2. Custom Reinforcement Learning Environment
Developed a complex, multi-agent (arena-style) trading environment based on Gymnasium. This environment accurately simulates real-market conditions, including:
* Slippage and transaction fees.
* Leverage and liquidation logic.
* A "Tom Level" system for curriculum learning (progressively increasing difficulty).
* A zero-sum reward function (`_calculate_arena_rewards`) to foster competitive agent behavior.

### 3. Model Architecture (ExtractorTransformer)
The agent's "eyes" use a hybrid **Transformer (TFM)** [user_prompt] and **TCN** architecture (`ExtractorTransformer`). This model processes both market history (`market_data`) and real-time account status (`account_data`) to generate a feature-rich state representation for the Actor-Critic policy.

### 4. Automated Training & HPO Pipeline
Created a scalable training pipeline using **Ray** to run multiple training experiments in parallel. This pipeline is fully integrated with **Optuna** to automatically search for the best-performing hyperparameters (e.g., learning rates, reward function weights).

### 5. Quantitative Evaluation & Backtesting
The framework includes a rigorous **backtesting** module (`evaluate_model`, `run_match`) to perform **model evaluation** on unseen validation data, measuring key financial metrics like Sharpe Ratio, PnL, and Win Rate.

---

## Project Showcase (Code Demo)

As the core trading logic and proprietary strategy parameters are a key part of this research, the full source code is not publicly available [user_prompt]. However, here are key architectural components demonstrating the project's capabilities.

### Showcase 1: Transformer-based Feature Extractor

This is the core `ExtractorTransformer` class, which combines TCNs and a **Transformer Encoder** [user_prompt] to process market data and account data into a single state vector for the RL agent.

```python
class ExtractorTransformer(BaseFeaturesExtractor):
    def __init__(self, obs_space, tcn_num_channels, tcn_kernel_size, tcn_dropout, tfm_nhead, tfm_nlayers, tfm_dim_feedforward, account_embedding_dim, **kwargs):
        tcn_output_dim = tcn_num_channels[-1]; super().__init__(obs_space, features_dim=tcn_output_dim + account_embedding_dim); mkt_shape, acc_dim = obs_space["market_data"].shape, obs_space["account_data"].shape[0]
        self.tcn = TemporalConvNet(mkt_shape[1], tcn_num_channels, kernel_size=tcn_kernel_size, dropout=tcn_dropout); self.pos_encoder = PositionalEncoding(tcn_output_dim, dropout=tcn_dropout, max_len=mkt_shape[0] + 1); encoder_layer = nn.TransformerEncoderLayer(d_model=tcn_output_dim, nhead=tfm_nhead, dim_feedforward=tfm_dim_feedforward, dropout=tcn_dropout, batch_first=True, activation='gelu'); self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=tfm_nlayers); self.acc_proc = nn.Sequential(nn.Linear(acc_dim, 64), nn.ReLU(), nn.Linear(64, account_embedding_dim), nn.Tanh())
    def forward(self, obs):
        mkt, acc = obs["market_data"], obs["account_data"]; tcn_out = self.tcn(mkt); pos_encoded = self.pos_encoder(tcn_out); tfm_out = self.transformer_encoder(pos_encoded); mkt_feat = tfm_out.mean(dim=1); acc_feat = self.acc_proc(acc); return torch.cat([mkt_feat, acc_feat], dim=1)
````

### Showcase 2: Automated HPO Pipeline (Optuna + Ray)

This is the `objective_worker` function that allows Optuna (for HPO) and Ray (for parallelism) to work together. It defines the entire "train-and-evaluate" job that can be distributed across multiple GPUs to find the optimal set of hyperparameters automatically.

```python
def objective_worker(trial: optuna.Trial, gpu_queue, training_params: dict,
                     train_data_arrays: dict, val_data_arrays: dict,
                     hparam_definer_func=None,
                     model_compiler_func=None,
                     objective_metric: str = "sharpe", seed_override: int = None,
                     load_checkpoint_path: str = None, save_checkpoint_path: str = None,
                     debug_mode: bool = False, detailed_log_path: str = None,
                     eval_episodes: int = 5):
    gpu_id = -1
    try:
        assert gpu_queue is not None, "Lỗi: gpu_queue không được cung cấp."
        gpu_id = gpu_queue.get(timeout=120)
        
        return objective(trial, gpu_id, training_params, 
                         train_data_arrays=train_data_arrays, 
                         val_data_arrays=val_data_arrays,
                         hparam_definer_func=hparam_definer_func,
                         model_compiler_func=model_compiler_func,
                         objective_metric=objective_metric, seed_override=seed_override,
                         load_checkpoint_path=load_checkpoint_path, save_checkpoint_path=save_checkpoint_path,
                         debug_mode=debug_mode, detailed_log_path=detailed_log_path,
                         eval_episodes=eval_episodes)
                         
    except optuna.exceptions.TrialPruned as e:
        raise e
    except Exception as e:
        print(f"---!!! LỖI NGHIÊM TRỌNG TRONG WORKER (GPU {gpu_id}, Seed {seed_override}) !!!---")
        traceback.print_exc()
        trial.set_user_attr("worker_error", traceback.format_exc())
        raise e
    finally:
        if gpu_id != -1 and gpu_queue is not None:
            gpu_queue.put(gpu_id)
```

### Showcase 3: Evaluation & Backtesting Results

*(**HƯỚNG DẪN:** Bạn hãy chạy một trong các ô Phân tích (ví dụ Ô 8.1, 9.7) và **chụp ảnh màn hình** biểu đồ PnL hoặc bảng xếp hạng kết quả. Dán ảnh vào đây.)*

<img width="913" height="442" alt="image" src="https://github.com/user-attachments/assets/078927f8-cfb3-4474-b07d-8b5424f4ab53" />

<img width="897" height="338" alt="image" src="https://github.com/user-attachments/assets/9853dc6f-3638-43f8-bbd4-de9e52e498a6" />

<img width="392" height="241" alt="image" src="https://github.com/user-attachments/assets/50d6818d-c166-43ad-b69b-7fdc6bf9139c" />
