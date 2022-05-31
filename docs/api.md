# API and Usage

As of now, there are three different patch interface classes:
- [](api_s1_header)
- [](api_s2_header)
- [](api_merged_header)

They all inherit from the same abstract base class and have very similar implementations.
If you would like to understand more about the implementation, check out the [](05_internal) document.

The most notable features are:
- The interface is very strict during the initialization
    - The data must have the correct shape (given by the Band type)
    - The _official_ name must be given
- If the bands are accessed, they are guaranteed to have a deterministic ordering
    - The bands are naturally sorted (`natsorted`)
- Extra metadata can be provided as key-word arguments, which will be stored as properties of the instance
- The data is internally stored as numpy arrays to allow framework-agnostic access
- Get stacked band data as a numpy array by calling `get_stacked_X_bands` methods
- The classes provide a quick interface to serialize and deserialize themselves via `pickle=v4`.
    - Useful for encoding/decoding the entire class as LMDB targets (see {{BenEncoder}} for details)

The [](api_merged_header) is a simple composition of the other Sentinel patch interfaces.
This interface was required to create a single LMDB file that contains both patches for a single key to maximize the read-performance.
