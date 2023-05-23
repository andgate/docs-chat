# Docs Chat

Small example project that creates a chatbot for documents!

#  Setup

## Python environment
(Instructions are for windows only.)

Install (conda)[https://docs.conda.io/en/latest/miniconda.html].

Create and activate our conda environment.
```
conda env create -f environment.yaml
conda activate docs-chat
```

If you add dependencies, the environment can be updated with:
```
conda env update -f environment.yaml
```

Also, don't forget to install this package locally so that importing works correctly.

```
pip install -e .
```

#  Running Servers

First, [setup a local qdrant instance.](https://qdrant.tech/documentation/quick_start/)

To run the chat server:
```
invoke start-server
```

To run the frontend
```
invoke start-frontend
```

Enjoy!