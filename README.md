# pyDISC
Time series jumps idealization with DISC

## Install
As of Dec 18, 2024, requires a slightly older Python version.
```shell
conda create -n pyDISC-env "python<3.12"
conda activate pyDISC-env
```
Requires a PyQt package. Should work with PySide6, PyQt6, or PyQt5.
```shell
pip install PySide6
```
Install latest development version:
```shell
pip install pyDISC@git+https://github.com/marcel-goldschen-ohm/pyDISC
```

## Use
```python
from qtpy.QtWidgets import QApplication
from pydisc import DISCO
app = QApplication()
widget = DISCO()

# input list of numpy arrays or DISC_Sequence objects
widget.data = ...

widget.show()
app.exec()
```
