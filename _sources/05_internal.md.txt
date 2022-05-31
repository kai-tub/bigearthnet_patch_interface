# Internals

Each patch interface class contains {class}`.Band` data.
The {class}`.Band` interface class is a simple base class that guarantess that the provided data is in the expected shape and has an associated name:

```{eval-rst}
.. autopydantic_model:: bigearthnet_patch_interface.band_interface.Band
```

The satellite specific bands are then used inside of a class that inherits from {class}`.BigEarthNetPatch`.
This class provides common functions that are used for all patches.
To guarantee deterministic ordering, the bands are always naturally sorted by their name.
To add flexibility the extra key-word arguments are stored as properties inside of each instance.
The recommended approach is to inherit from this class and to call `super().__init__` to ensure that everything is compatible.

```{eval-rst}
.. autoclass:: bigearthnet_patch_interface.patch_interface.BigEarthNetPatch
    :members:
```
