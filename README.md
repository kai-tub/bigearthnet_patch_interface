# BigEarthNet Patch Interface
> A simple interface class that includes all the relevant information about BigEarthNet patches.

[![Tests](https://img.shields.io/github/workflow/status/kai-tub/bigearthnet-patch-interface/CI?color=dark-green&label=%20Tests)](https://github.com/kai-tub/bigearthnet_patch_interface//actions/workflows/main.yml)
[![MyPy Type Checker](https://img.shields.io/badge/%20type_checker-mypy-%231674b1?style=flat&color=dark-green)](http://mypy-lang.org/)
[![License](https://img.shields.io/pypi/l/bigearthnet-patch-interface?color=dark-green)](https://github.com/kai-tub/bigearthnet_patch_interface//blob/main/LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/bigearthnet-patch-interface)](https://pypi.org/project/bigearthnet-patch-interface)
[![PyPI version](https://img.shields.io/pypi/v/bigearthnet-patch-interface)](https://pypi.org/project/bigearthnet-patch-interface)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/bigearthnet-patch-interface?color=dark-green)](https://anaconda.org/conda-forge/bigearthnet-patch-interface)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)
[![Auto Release](https://img.shields.io/badge/release-auto.svg?colorA=888888&colorB=blueviolet&label=auto&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAYAAACNiR0NAAACzElEQVR4AYXBW2iVBQAA4O+/nLlLO9NM7JSXasko2ASZMaKyhRKEDH2ohxHVWy6EiIiiLOgiZG9CtdgG0VNQoJEXRogVgZYylI1skiKVITPTTtnv3M7+v8UvnG3M+r7APLIRxStn69qzqeBBrMYyBDiL4SD0VeFmRwtrkrI5IjP0F7rjzrSjvbTqwubiLZffySrhRrSghBJa8EBYY0NyLJt8bDBOtzbEY72TldQ1kRm6otana8JK3/kzN/3V/NBPU6HsNnNlZAz/ukOalb0RBJKeQnykd7LiX5Fp/YXuQlfUuhXbg8Di5GL9jbXFq/tLa86PpxPhAPrwCYaiorS8L/uuPJh1hZFbcR8mewrx0d7JShr3F7pNW4vX0GRakKWVk7taDq7uPvFWw8YkMcPVb+vfvfRZ1i7zqFwjtmFouL72y6C/0L0Ie3GvaQXRyYVB3YZNE32/+A/D9bVLcRB3yw3hkRCdaDUtFl6Ykr20aaLvKoqIXUdbMj6GFzAmdxfWx9iIRrkDr1f27cFONGMUo/gRI/jNbIMYxJOoR1cY0OGaVPb5z9mlKbyJP/EsdmIXvsFmM7Ql42nEblX3xI1BbYbTkXCqRnxUbgzPo4T7sQBNeBG7zbAiDI8nWfZDhQWYCG4PFr+HMBQ6l5VPJybeRyJXwsdYJ/cRnlJV0yB4ZlUYtFQIkMZnst8fRrPcKezHCblz2IInMIkPzbbyb9mW42nWInc2xmE0y61AJ06oGsXL5rcOK1UdCbEXiVwNXsEy/6+EbaiVG8eeEAfxvaoSBnCH61uOD7BS1Ul8ESHBKWxCrdyd6EYNKihgEVrwOAbQruoytuBYIFfAc3gVN6iawhjKyNCEpYhVJXgbOzARyaU4hCtYizq5EI1YgiUoIlT1B7ZjByqmRWYbwtdYjoWoN7+LOIQefIqKawLzK6ID69GGpQgwhhEcwGGUzfEPAiPqsCXadFsAAAAASUVORK5CYII=)](https://github.com/intuit/auto)

A common issue when using a BigEarthNet archive is that the code to load a patch is
- Slow
- The necessary libraries to load the data have complex dependencies and cause issues with popular deep-learning libraries
  - The issue is often caused by a binary mismatch between the underlying `numpy` versions
- Hard to understand how to access the optimized data

A popular approach is to use the key-value storage [LMDB](https://lmdb.readthedocs.io/en/release/).
The patch names are set as keys, and the value is _somehow_ encoded.
Decoding the values is a common source of bugs when different deep-learning libraries are used.

The goal of this repository is to alleviate this issue.
The actual image data will be encoded as `numpy` arrays to support the most popular deep-learning libraries.
Usually, these arrays can be loaded without copying the underlying data.

The provided patch interface will define a Python class containing the relevant bands, encoded as an `np.ndarray`, and may include some metadata.
The interface class allows for fast introspection, validation, and data loading.

It is easier to convert this intermediate data format to a more deep-learning optimized format or use the class itself for embedding in a key-value storage format like LMDB.
Please refer to the official {{BenEncoder}} documentation for details on how to _generate_ various output formats.

In general, the encoding pipeline is as follows:
1. Convert the BigEarthNet patches into numpy arrays
2. Use these arrays to initialize the interface class
3. Provide any additional metadata information to the interface
4. Serialize/convert the data to the desired output format

The decoder should be as simple as possible with minimal dependencies.
Ideally, the correct usage of the decoder should also be depicted in the corresponding {{BenEncoder}} section.

:::{note}
This interface is tightly coupled with the {{BenEncoder}} libary!

The main reason for providing this interface as a _standalone_ package is to minimize the required dependencies of a user that wants to load/use already encoded patches.
Creating the interface from the _raw_ data requires heavy libraries that often cause binary conflicts.
The interface class can be installed as a standalone library and only requires the (easier to install) _decoder_ libraries.

:::
