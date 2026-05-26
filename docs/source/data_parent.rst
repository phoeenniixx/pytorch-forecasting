Data
====

.. currentmodule:: pytorch_forecasting.data

Currently, PyTorch Forecasting offers two implementations for the data pipeline:

- The **v1 Data Pipeline** is built entirely around the :py:class:`~timeseries.TimeSeriesDataSet` class. This unified class handles data definition, scaling, and transformation, and is used to directly instantiate v1 models using their ``.from_dataset()`` methods present in the models.

- The **v2 Data Pipeline** introduces a decoupled workflow that separates data definition from loading and pre-processing:

  - :py:class:`~timeseries.TimeSeries` handles data ingestion, structural definitions, and metadata mapping.
  - :py:mod:`~data_module` handles data loading, batching, and preprocessing steps.

  .. warning::
    Please note that the v2 components are currently under active development (beta). Use this API with caution as interfaces may undergo changes before the final release.


.. toctree::
    :maxdepth: 2

    Data-v1 <data_v1>
    Data-v2 <data_v2>
