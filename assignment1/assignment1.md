
# Assignment 1

## Setup

Navigate to the src directory

```bash
cd assignment1/src
```

Create the virtual environment and install dependencies

```bash
uv sync
```

Activate the environment

```bash
source .venv/bin/activate
```

Go back to the root of the assignment

```bash
cd ..
```

## To run

Make sure to be in the assignment1 directory

```bash
pwd # should end with assignment1
```

Follow the setup instructions above to create and activate the virtual environment, then run the main file as a module (allows relative imports)

```bash
python -m src.main
```