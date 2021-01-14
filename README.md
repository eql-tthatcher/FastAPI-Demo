# FastAPI Demo

To set up the Conda environment:

```makefile
make environment
```

Next, fit the model and generate an artifact with the following recipe:

```makefile
make model
```

Now that the model artifact is ready, the service can be run locally:

```makefile
make debug
```
