Installation
============

Basic Installation
------------------

Install the core package using pip:

.. code-block:: bash

    pip install pl-fuzzy-frame-match

Or using Poetry:

.. code-block:: bash

    poetry add pl-fuzzy-frame-match

Installation for Large Datasets
-------------------------------

Approximate matching support (``polars-simed``) is installed with the package. It is used automatically
for large datasets (>100M potential matches), so no extra install step is needed.

Requirements
------------

* Python >= 3.10
* Polars >= 1.39.0 (including 2.x)
* polars-distance >= 0.5.3, < 0.6
* polars-simed >= 0.4.0

Verify Installation
-------------------

.. code-block:: python

    import pl_fuzzy_frame_match
    print(pl_fuzzy_frame_match.__version__)
