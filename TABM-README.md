# TABM Implementation for Spaceship Titanic
Contributer: Elias Ruud Aronsen

This project contains my implementation of **TABM** (Tabular Multiple Predictions), a simple but powerful method for deep learning on tabular data.

TABM makes a single MLP behave like an ensemble of multiple MLPs by sharing most parameters while producing multiple independent predictions per input. Inspired by BatchEnsemble, but specifically tuned for tabular tasks, TABM achieves better generalization, faster training, and smaller model sizes compared to traditional deep ensembles or transformer-style models.

#### Key ideas:
- Multiple predictions per sample, trained together, we choose the amount by defining **k**, which is a tunable parameter.

- Heavy weight sharing to keep it efficient and faster. Most model weights are shared across submodels, keeping the architecture lightweight and efficient.

- Even though individual predictions may be weak, the aggregated output benefits from strong generalization due to implicit ensembling.

## My Implementation

For this implementation, I build on the original TABM PyTorch codebase provided by the authors.  
The workflow is as follows:

1. **Data Preparation**  
   Preprocessed data (prepared in `1-EDA-and-preprocessing.ipynb`) is loaded and split into training and validation sets using stratified sampling to preserve the class distribution.

2. **Model Setup**  
   A `setup` function defines the TABM model, optimizer (AdamW), and loss function (CrossEntropyLoss), with parameters passed dynamically to allow us to use tuning methods like Optuna.

3. **Hyperparameter Optimization**  
   Instead of manually tuning hyperparameters, I use **Optuna**, a smarter hyperparameter search framework.
   - 50 trials are run.
   - Optuna suggests new parameters based on previous results.
   - Bad trials are automatically pruned early to save computation time.

4. **Training Loop**  
   For each trial:
   - The model is trained using mini-batches with early stopping based on validation loss. I sat the patience to 20 after a few tries finding that any higher just wastes time.
   - Average training loss, validation loss, and accuracy are monitored at each epoch.

5. **Final Model Selection and Evaluation**  
   After the search, the best hyperparameter combination (based on validation accuracy) is selected.  
   The model is then retrained on the full training dataset and evaluated against other models used in the project to assess its performance.


#### Info:
You can see the full implementation in the **3-TABM.ipynb** notebook.

This work builds on code from the authors' official repository: [TabM GitHub Repo](https://github.com/yandex-research/tabm).

You can read the paper on the novel model here: [TabM Paper](https://arxiv.org/abs/2410.24210).