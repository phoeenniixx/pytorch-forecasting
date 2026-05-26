Models
======
.. currentmodule:: pytorch_forecasting

Currently, we have two collections of models:

- The **:doc:`v1 Models <models>`** that use a ``.from_dataset()`` method for each model that
takes a :py:class:`~data.timeseries.TimeSeriesDataSet` and additional parameters
that cannot directly derived from the dataset such as, e.g. ``learning_rate`` or
``hidden_size``.

- The **:doc:`v2 Models <models_v2>`** that use a decoupled pipeline of:
   - :py:class:`~data.timeseries.TimeSeries` for data ingestion and
   - :py:mod:`~data.data_module` for pre-processing and data loading
   - ``model_pkg`` (unique for all models) class for simple ``fit`` - ``predict`` like
     interface
  You can pass the additional parameters that cannot directly derived from the data
  (e.g.  ``learning_rate`` or ``hidden_size``), directly to the ``pkg`` class.
  For more info, have a look at :doc:`API v2 <api_v2>`.

  .. warning::
    Please note that the V2 modules are currently in active-development and is in beta right now, so please use this API with caution.


.. toctree::
    :maxdepth: 2

    Models-v1 <models>
    Models-v2 <models_v2>
