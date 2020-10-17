# Development of a Sepsis Early Warning Indicator

## To run code:

- Install Docker (will need admin permission): https://hub.docker.com/editions/community/docker-ce-desktop-windows

- Build docker image:
```
docker build -t physionet .
```

- Create test directories: 1) Copy some of the training data into the a test input directory and 2) create an empty output directory. This is whapythont my test directories look like:
```
C:\Users\PouPromC\Projects\physionet-2019\test_data (abstract-submission -> origin)
λ ls
input_dir/  output_dir/
C:\Users\PouPromC\Projects\physionet-2019\test_data (abstract-submission -> origin)
λ ls input_dir\
p000001.psv
C:\Users\PouPromC\Projects\physionet-2019\test_data (abstract-submission -> origin)
λ ls output_dir\
```


- Mount the test directories to the Docker countainer and launch the container.

```
docker run -v PATH_TO_INPUT_DIRECTORY\:/physionet2019/input_directory -v PATH_TO_OUTPUT_DIRECTORY:/physionet-2019/output_dir -it physionet bash
```

- Run the `driver.py` code.
```
python driver.py input_directory output_dir
```
