Data V2
=======

.. currentmodule:: pytorch_forecasting

Loading and managing time series data for deep learning can be complex, especially when handling varying sequence lengths, multiple covariates, and categorical encodings.

In API-v2, the data pipeline follows a strict two-layer architecture: the **D1 Layer (Dataset)** and the **D2 Layer (DataModule)** to maintain "separation of responsibilities".

- **D1 Layer (Dataset)** ingests the raw data and turn it into ``torch`` tensors
- **D2 Layer (DataModule)** performs the pre-processing and the data loading


D1 Layer: Dataset
-----------------

The D1 Layer is the foundational data ingestion layer. Its primary responsibilities are to accept raw tabular data (e.g., pandas DataFrames), convert the raw data into PyTorch tensors, and extract base-level metadata such as static variables and basic time series properties.

Unlike the V1 dataset, the D1 layer does not handle complex preprocessing or batching logic, keeping it lightweight and highly modular.

.. autoclass:: pytorch_forecasting.data.timeseries._timeseries_v2.TimeSeries
   :noindex:
   :members: __init__


D2 Layer: DataModule
--------------------

The D2 Layer sits on top of D1 and is implemented as a PyTorch Lightning ``LightningDataModule``. This layer is responsible for the heavier lifting:

* **Preprocessing:** Applying normalizers and encoders to the data.
* **Batching:** Creating and managing the ``train_dataloader``, ``val_dataloader``, and ``test_dataloader``.
* **Model Initialization Metadata:** Dynamically collecting necessary architectural information (such as the number of categorical variables, embedding sizes, and vocabulary states) required to properly instantiate the Forecasting models in the Model Layer.

.. autoclass:: pytorch_forecasting.data._tslib_data_module.TslibDataModule
   :noindex:
   :members: __init__


API Reference
-------------

See the detailed API documentation for the V2 data classes below:

.. currentmodule:: pytorch_forecasting

.. autosummary::
   :toctree: api/
   :template: custom-module-template.rst

   pytorch-forecasting.data.encoders.EncoderNormalizer
   pytorch-forecasting.data.encoders.GroupNormalizer
   pytorch-forecasting.data.encoders.MultiNormalizer
   pytorch-forecasting.data.encoders.NaNLabelEncoder
   pytorch-forecasting.data.encoders.TorchNormalizer
   pytorch-forecasting.data.samplers.TimeSynchronizedBatchSampler
   pytorch-forecasting.data.samplers.GroupedSampler
   pytorch-forecasting.data.timeseries._timeseries_v2.TimeSeries
   pytorch-forecasting.data.data_module.EncoderDecoderTimeSeriesDataModule
   pytorch-forecasting.data._tslib_data_module.TslibDataModule
