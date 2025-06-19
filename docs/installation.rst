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

For optimal performance with large datasets (>100M potential matches), install with approximate matching support:

.. code-block:: bash

    pip install pl-fuzzy-frame-match polars-simed

Requirements
------------

* Python >= 3.9
* Polars >= 1.8.2, < 2.0.0
* polars-distance ~= 0.4.3
* polars-simed >= 0.3.4 (optional, for large datasets)

Verify Installation
-------------------

.. code-block:: python

    import pl_fuzzy_frame_match
    print(pl_fuzzy_frame_match.__version__)