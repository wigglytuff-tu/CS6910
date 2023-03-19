# CS6910
Assignment-1: Write your own backpropagation code and keep track of your experiments using wandb.ai

## Instructions to run
First must install: ``` pip install wandb ```<br>
Then run: ```python train.py```
Additional arguments can be given and usage can be identified by executing ```python train.py --help```
Other dependencies include: numpy and tendorflow.keras libraries
## File Structure
```train.py``` consists of three fuctions:
- train()--> to run sweep
- get_confusion_matrix()--> Question-7
- get_images()--> Question-1
By default we run train function, but other two functions can be executed as well

```template.py```
This file consists of template classes to implement differentiable classes sourced from internet.

```helper.py```
This file consists of all the helper functions for activations, optimizers and Model.




