## Experiment

This experiment is for training student model from KeFlow and measure performance.
<br>
<br>

- Train Knowledge Extraction Normalizing-Flow(KeFlow) from classifier
<br>

```python
python train_keflow.py
```

<br>

- Re-generate distribution of training data from trained KeFlow
<br>

```python
python generate_data.py
```

- Train Teacher-Student model with re-generated training data
<br>

```python
python train_student.py
```
