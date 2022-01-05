# BigEarthNet Patch Interface

A common issue when using a BigEarthNet archive is that the code to load a patch is
- Slow
- The necessary libraries to load the data have complex dependencies and cause issues with popular deep-learning libraries
  - Hint, the issue is often caused by a binary mismatch between the underlying `numpy` versions
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
