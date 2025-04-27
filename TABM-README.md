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

5. **Cross-Validation for Final Evaluation**  
   After selecting the best hyperparameter configuration based on validation accuracy, I further evaluated the model using five-fold stratified cross-validation.  
   - The full dataset was split into five folds while maintaining class distribution.
   - For each fold, the model was trained on four folds and validated on the remaining fold.
   - Validation accuracies from all folds were collected, and the mean validation accuracy was reported as a more robust performance estimate.

6. **Final Model Training**  
   After cross-validation, a final TABM model was trained from scratch on the full dataset using the best hyperparameters, ensuring it had access to all available data for maximum performance.

## Final Result

- The final model achieved a validation accuracy of **0.8470**, which is a strong result during training.
- However, the final Kaggle leaderboard score for the TABM model was **0.79541**, indicating signs of overfitting to the training data.
- The five-fold cross-validation gave a mean validation accuracy of **0.7927**, suggesting that the true generalization performance of the model is slightly lower than the single validation split suggested.

I hypothesize that the model's complexity — and the high capacity of MLP architectures in general — made it prone to overfitting given the relatively small size of the Spaceship Titanic dataset.  
Despite the regularization strategies used (dropout, weight decay, batch normalization), the model likely still overfit subtle patterns in the training data that did not generalize well to unseen test data.


#### Info:
You can see the full implementation in the **3-TABM.ipynb** notebook.

This work builds on code from the authors' official repository: [TabM GitHub Repo](https://github.com/yandex-research/tabm).

You can read the paper on the novel model here: [TabM Paper](https://arxiv.org/abs/2410.24210).