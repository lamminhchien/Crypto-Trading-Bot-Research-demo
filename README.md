# AI-Powered Quantitative Trading Research Framework

This repository serves as a research framework for "Tom-AI," a project focused on applying advanced AI models for quantitative trading in Crypto and Forex markets.

The primary goal is not just to build a single bot, but to create a robust pipeline for **backtesting**, **evaluating**, and **benchmarking** different AI-driven trading strategies. This project is heavily focused on the "Research" aspect of quantitative analysis.

## 🚀 Core Technologies & Concepts

* **AI Framework:** PyTorch
* **Core Architecture:** **Transformer (TFM)** [user_prompt] & Temporal Convolutional Networks (TCN) for feature extraction from market data.
* **Strategy Model:** Reinforcement Learning (Soft Actor-Critic) to train an agent that makes trading decisions.
* **MLOps Pipeline:** **Optuna** for Hyperparameter Optimization (HPO) and **Ray** for distributed, multi-GPU training.
* **Experimental Research:** Mamba (SSM) integration for benchmarking against Transformer-based models.
* **Data Science Stack:** Python, Pandas, NumPy, Scikit-learn.

## 📈 Methodology

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

## 📊 Project Showcase (Demo)

As the core trading logic and proprietary strategy parameters are a key part of this research, the full source code is not publicly available [user_prompt]. However, here is a showcase of the project's capabilities and architecture.

### Showcase 1: Transformer-based Feature Extractor
This is the core `ExtractorTransformer` class, which combines TCNs and a **Transformer Encoder** [user_prompt] to process market data.

*(**HƯỚNG DẪN:** Bạn hãy chụp ảnh màn hình đoạn code class `ExtractorTransformer` trong file notebook của bạn và dán vào đây)*
`[DÁN ẢNH CHỤP MÀN HÌNH CODE CỦA BẠN VÀO ĐÂY]`

### Showcase 2: Automated HPO Pipeline (Optuna + Ray)
This is the `objective_worker` function that allows Optuna and Ray to work together, automatically training and evaluating hundreds of different model variations across multiple GPUs to find the optimal strategy.

*(**HƯỚNG DẪN:** Bạn hãy chụp ảnh màn hình hàm `objective_worker` của bạn và dán vào đây)*
`[DÁN ẢNH CHỤP MÀN HÌNH CODE CỦA BẠN VÀO ĐÂY]`

### Showcase 3: Evaluation & Backtesting Results
Example of evaluation results from a training run, showing key performance metrics.

*(**HƯỚNG DẪN:** Bạn hãy chạy một trong các ô Phân tích (ví dụ Ô 8.1, 9.7) và chụp ảnh màn hình biểu đồ PnL hoặc bảng xếp hạng kết quả. Dán ảnh vào đây.)*
`[DÁN ẢNH CHỤP MÀN HÌNH KẾT QUẢ/BIỂU ĐỒ CỦA BẠN VÀO ĐÂY]`
