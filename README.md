# BigEarthNet Patch Interface
[![Tests](https://img.shields.io/github/workflow/status/kai-tub/bigearthnet_patch_interface/CI?color=dark-green&label=%20Tests)](https://github.com/kai-tub/bigearthnet_patch_interface/actions/workflows/main.yml)
[![License](https://img.shields.io/pypi/l/bigearthnet_patch_interface?color=dark-green)](https://github.com/kai-tub/bigearthnet_patch_interface/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/bigearthnet-patch-interface.svg)](https://pypi.org/project/bigearthnet-patch-interface/)
[![Auto Release](https://img.shields.io/badge/release-auto.svg?colorA=888888&colorB=9B065A&label=auto&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAYAAACNiR0NAAACzElEQVR4AYXBW2iVBQAA4O+/nLlLO9NM7JSXasko2ASZMaKyhRKEDH2ohxHVWy6EiIiiLOgiZG9CtdgG0VNQoJEXRogVgZYylI1skiKVITPTTtnv3M7+v8UvnG3M+r7APLIRxStn69qzqeBBrMYyBDiL4SD0VeFmRwtrkrI5IjP0F7rjzrSjvbTqwubiLZffySrhRrSghBJa8EBYY0NyLJt8bDBOtzbEY72TldQ1kRm6otana8JK3/kzN/3V/NBPU6HsNnNlZAz/ukOalb0RBJKeQnykd7LiX5Fp/YXuQlfUuhXbg8Di5GL9jbXFq/tLa86PpxPhAPrwCYaiorS8L/uuPJh1hZFbcR8mewrx0d7JShr3F7pNW4vX0GRakKWVk7taDq7uPvFWw8YkMcPVb+vfvfRZ1i7zqFwjtmFouL72y6C/0L0Ie3GvaQXRyYVB3YZNE32/+A/D9bVLcRB3yw3hkRCdaDUtFl6Ykr20aaLvKoqIXUdbMj6GFzAmdxfWx9iIRrkDr1f27cFONGMUo/gRI/jNbIMYxJOoR1cY0OGaVPb5z9mlKbyJP/EsdmIXvsFmM7Ql42nEblX3xI1BbYbTkXCqRnxUbgzPo4T7sQBNeBG7zbAiDI8nWfZDhQWYCG4PFr+HMBQ6l5VPJybeRyJXwsdYJ/cRnlJV0yB4ZlUYtFQIkMZnst8fRrPcKezHCblz2IInMIkPzbbyb9mW42nWInc2xmE0y61AJ06oGsXL5rcOK1UdCbEXiVwNXsEy/6+EbaiVG8eeEAfxvaoSBnCH61uOD7BS1Ul8ESHBKWxCrdyd6EYNKihgEVrwOAbQruoytuBYIFfAc3gVN6iawhjKyNCEpYhVJXgbOzARyaU4hCtYizq5EI1YgiUoIlT1B7ZjByqmRWYbwtdYjoWoN7+LOIQefIqKawLzK6ID69GGpQgwhhEcwGGUzfEPAiPqsCXadFsAAAAASUVORK5CYII=)](https://github.com/intuit/auto)
<!-- [![Conda Version](https://img.shields.io/conda/vn/conda-forge/bigearthnet-patch-interface?color=dark-green)](https://anaconda.org/conda-forge/bigearthnet-patch-interface) -->


A common issue when using a BigEarthNet archive is that the code to load a patch is
- Slow
- The necessary libraries to load the data have complex dependencies and cause issues with popular deep-learning libraries
  - The issue is often caused by a binary mismatch between the underlying `numpy` versions
- Hard to understand how to access the optimized data

A popular approach is to use the key-value storage [LMDB](https://lmdb.readthedocs.io/en/release/).
The patch names are set as a key, and the value is _somehow_ encoded.
Decoding the values is a common source of bugs when different deep-learning libraries are used.

The goal of this repository is to alleviate this issue.
To support most popular deep-learning libraries, the actual image data will be encoded as `numpy` arrays.
Usually, these arrays can be loaded without copying the underlying data.

The provided patch interface will define a Python class containing the 12 bands, encoded as an `np.ndarray`, and may include some metadata.
The class allows for fast introspection, validation, and data loading.

In general, the encoding pipeline is as follows:
1. To convert the BigEarthNet patches into numpy arrays
2. Use these arrays to initialize the interface class
3. Provide any additional metadata information to the interface
4. Pickle the instance
5. Store the instance to an LMDB database as a value (usually, the name of the patch is used as the key)

Only the LMDB database and the interface class are required to load the pre-converted data.
This repository only contains the interface to minimize the required dependencies to unpickle the data.

Please be aware that pickle is **insecure**, and pickled data should **never** be trusted.
For a detailed review on `pickle`, see the following post of [synopsys on pickling](https://www.synopsys.com/blogs/software-security/python-pickling/).
The primary benefit of pickle is that it comes with Python (requires no heavy dependencies), is blazingly fast, and allows for efficient serialization of various data types.

Do not forget to restrict write access to the pickled files when using them in a shared environment!
