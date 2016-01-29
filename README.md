# mmln

Machine learning tools for massively multi-labeled networks

# Installation

You can install this library by running the following command inside the
top-level directory

```
pip install .
```

# Representation

The mmln library represents networks using [networkx](https://networkx.github.io/).
Label information is stored in dictionaries using three node attributes, `mmln.OBSVS`,
`mmln.TARGETS`, and `mmln.TRUTH`. `mmln.OBSVS` stores label values that are observed,
i.e. given. For example,

python
```
network.node['Node 1'][mmln.OBSVS] = {'Label 1': 1, 'Label 2': 0}
```

says that Label 1 is observed with a value of 1, i.e. True and Label 2 is observed with
a value of 0, i.e. False.

Likewise, `mmln.TARGETS` stores predictions, such as from a prediction algorithm. Putting
an entry for a label in a `mmln.TARGETS` dictionary attribute indicates to prediction
algorithms that they should make a prediction for that node-label pair. The initial value
stored for the target does not matter. Prediction algorithms will overwrite it.

`mmln.TRUTH` is for storing true values for the corresponding targets on the node. 