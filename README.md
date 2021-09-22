# Multi-Head attention for motion prediction

We define in this git a model implementing multihead attention for motion prediction.
Take a loot at our [project report](multihead_attention_report.pdf)

# How to run

To run the model, simply run the following commands, inside the main project directory:

```python3 train.py --n_epochs=20000 --bs_train=256```

To evaluate the model, run

```python3 evaluate.py --model_id=XXX```

where XXX is the trained models id.
