# Alphafold3

## Install

```bash
git clone https://github.com/google-deepmind/alphafold3
cd alphafold3
# download database (very large)
bash ./fetch_databases.sh ./database
# create environment
conda create -p ./env python=3.11
conda activate ./env
conda install gxx=12 zlib cmake -c conda-forge
# install hmmer
mkdir hmmer_build hmmer
cd hmmer_build
wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz
tar -xvf hmmer-3.4.tar.gz
cd hmmer-3.4
./configure --prefix /absolute/path/to/alphafold3/hmmer
make -j32
make install
```

Add contents in `CMakeLists.txt`:

```txt
...
include(FetchContent)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(ABSL_PROPAGATE_CXX_STD ON)
# the following line is added
set(ZLIB_INCLUDE_DIR "/absolute/path/to/alphafold3/env/include/")
...
```

also zlib in **target_link_libraries** (the last line is added):

```txt
target_link_libraries(
  cpp
  PRIVATE absl::check
          absl::flat_hash_map
          absl::node_hash_map
          absl::strings
          absl::status
          absl::statusor
          absl::log
          pybind11_abseil::absl_casters
          Python3::NumPy
          dssp::dssp
          cifpp::cifpp
          ${ZLIB_LIBRARIES})
```

Then get back to the root directory of alphafold3 and use pip to install the dependencies:

```bash
pip3 install --no-deps .
pip3 install tqdm
```

Build data for CCD:

```bash
./env/bin/build_data
```

Modify the `REPO_DIR` environment to the absolute path of your alphafold3 folder in the provided script `alphafold3_predict.sh` under this repo, and use it for inference.

Example with disabled MSA and template search:

```bash
./alphafold3_predict.sh --json_path ./example_input.json --out_dir ./example_output
```

If you want to enable MSA for a certain chain, just delete the two lines for the chain:

```json
"unpairedMsa": "",
"pairedMsa": "",
```

To enable template, just delete the following line:

```json
"templates": []
```