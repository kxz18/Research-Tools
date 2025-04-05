An example of implementing an MLP to fit sin(x)

1. Create a dataset with registration: `./data/sin_example_dataset.py`. Import it in `./data/__init__.py`.

2. Create a model with registration: `./models/sin_example_mlp.py`, and import it in `./models/__init__.py`.

3. Create a trainer with registration: `./trainer/sin_example_trainer.py`, and import it in `./trainer/__init__.py`.

4. Create a config: `./configs/sin_example_config.yaml`.

5. Run training: `GPU=0 bash scripts/train.sh configs/sin_example_config.yaml`. If multiple GPU are specified (e.g. `GPU=0,1`), distributed parallel training is enabled.
