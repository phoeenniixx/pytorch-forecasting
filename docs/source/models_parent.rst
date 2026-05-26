Models
======
.. currentmodule:: pytorch_forecasting

Currently, PyTorch Forecasting offers two architectural approaches for models:

- The **:doc:`v1 Models <models>`** that use a ``.from_dataset()`` method for each model that takes a :py:class:`~data.timeseries.TimeSeriesDataSet` and additional parameters that cannot directly derived from the dataset such as, e.g. ``learning_rate`` or ``hidden_size``.

- The **:doc:`v2 Models <models_v2>`** utilize a modular pipeline separated into distinct components:
   - :py:class:`~data.timeseries.TimeSeries` for data ingestion.
   - Different types of :py:mod:`~data.data_module` for pre-processing and data loading. Each ``data_module`` is compatible with a specific type of models. Please look at :doc:`Data v2<data_v2>` and :doc:`Model v2 <models_v2>` documentation for more info
   - A unique ``model_pkg`` class providing a ``sklearn``-like ``fit`` and ``predict`` interface.

  You can pass the additional parameters that cannot directly derived from the data (e.g.  ``learning_rate`` or ``hidden_size``), directly to the ``pkg`` class.

  For more info, have a look at :doc:`API v2 <api_v2>`.

  .. warning::
    Please note that the V2 modules are currently in active-development and is in beta right now, so please use this API with caution.


.. toctree::
    :maxdepth: 2

    Models-v1 <models>
    Models-v2 <models_v2>
