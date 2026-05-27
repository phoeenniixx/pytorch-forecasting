Models
======
.. currentmodule:: pytorch_forecasting

Currently, PyTorch Forecasting offers two architectural approaches for models:

- The :doc:`v1 Models <models>` that use a ``.from_dataset()`` method for each model that takes a :py:class:`~data.timeseries.TimeSeriesDataSet` and additional parameters that cannot directly derived from the dataset such as, e.g. ``learning_rate`` or ``hidden_size``.

- The :doc:`v2 Models <models_v2>` utilize a modular pipeline separated into distinct components:
   - :py:class:`~data.timeseries.TimeSeries` for data ingestion.
   - Different types of :py:mod:`~data.data_module` for pre-processing and data loading. Each ``data_module`` is compatible with a specific type of models. Please look at :doc:`Data v2<data_v2>` and :doc:`Model v2 <models_v2>` documentation for more info
   - A unique ``model_pkg`` class providing a ``sklearn``-like ``fit`` and ``predict`` interface.

  You can pass the additional parameters that cannot directly derived from the data (e.g.  ``learning_rate`` or ``hidden_size``), directly to the ``pkg`` class.

  For more info, have a look at :doc:`API v2 <api_v2>`.

  .. warning::
    Please note that the V2 modules are currently in active-development and is in beta right now, so please use this API with caution.

Choosing Between V1 and V2
--------------------------

While V2 represents the future of PyTorch Forecasting, it is currently a Work in Progress (WiP). You should choose your API based on your current project requirements:

* **When to use V1:** If you are building critical production workflows, require mixed data types (categorical and continuous targets), or rely heavily on multi-target forecasting. V1 is stable and fully feature-complete.
* **When to try V2:** If you are starting a new project, exploring the library, or want a cleaner, more modular codebase. **Note:** V2 currently only handles numeric data, and multi-target forecasting is not yet correctly handled. We highly encourage you to test V2 and provide feedback on our GitHub repository, but do not push it into production environments that require those missing features just yet.


Selecting an Architecture
-------------------------

Criteria for selecting a forecasting architecture depend heavily on the use-case. There are multiple selection criteria you should take into account regardless of whether you are using the V1 or V2 API. Here is an overview of the key considerations:

**v1 models**

.. model-overview-v1::

**v2 models**

.. model-overview-v2::

Size and type of available data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One should particularly consider five criteria when evaluating data:

**Availability of covariates**
If you have covariates—variables in addition to the target variable itself that hold information about the target—then your case will benefit from a model that can accommodate them. A model that *cannot* use covariates is ``NBeats``.

**Length of timeseries**
The length of time series has a significant impact on which model will work well. Most models are created and tested on very long timeseries, while in practice, short (or a mix of short and long) timeseries are often encountered. A model that can leverage covariates well, such as the ``TemporalFusionTransformer``, will typically perform better than other models on short timeseries. Making cold-start predictions solely based on static covariates (making predictions without observed history) is supported by the ``TemporalFusionTransformer``, but does not always work tremendously well.

**Number of timeseries and their relation to each other**
If your time series are related to each other (e.g., all sales of products of the same company), a model that can learn relations between the timeseries can improve accuracy. Note that only models that can process covariates can learn relationships between different timeseries. If the timeseries denote different entities or exhibit very similar patterns across the board, a model such as ``NBeats`` will not work as well. If you have only one or very few timeseries, they should be very long in order for a deep learning approach to work well.

**Type of prediction task**
Not every model can do regression, classification, or handle multiple targets. Some are exclusively geared towards a single task. For example, ``NBeats`` can only be used for regression on a single target without covariates, while the ``TemporalFusionTransformer`` supports multiple targets and even heterogeneous targets where some are continuous variables and others categorical (regression and classification at the same time). ``DeepAR`` can handle multiple targets but only works for regression tasks.

For long forecast horizon forecasts, ``NHiTS`` is an excellent choice as it uses interpolation capabilities.

Supporting uncertainty
~~~~~~~~~~~~~~~~~~~~

Not all models support uncertainty estimation. Those that do might do so in different fashions. Non-parametric models provide forecasts that are not bound to a given distribution, while parametric models assume that the data follows a specific distribution.

Parametric models will be a better choice if you know how your data (and potentially error) is distributed. However, if you are missing this information or cannot make an educated guess that matches reality, the model's uncertainty estimates will be adversely impacted. In this case, a non-parametric model will do much better.

``DeepAR`` is an example of a parametric model, while the ``TemporalFusionTransformer`` can output quantile forecasts that can fit any distribution.

Computational requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~

Some models have simpler architectures and fewer parameters than others, which can lead to significantly different training times. However, this is not a general rule. Because the data for a sample for timeseries models is often far smaller than it is for computer vision or language tasks, GPUs are often underused, and increasing the width of models can be an effective way to fully use a GPU. This can increase the speed of training while also improving accuracy.

The other path to pushing utilization of a GPU up is increasing the batch size. However, increasing the batch size can adversely affect the generalization abilities of a trained network.

Also, take into account that computational resources are mainly necessary for inference/prediction. The upfront task of training a model will require developer time, but might be only a small part of the total computational costs over the lifetime of a model.

The ``TemporalFusionTransformer`` is a rather large model, whereas ``NBeats`` or ``NHiTS`` are highly efficient. Autoregressive models such as ``DeepAR`` might be quick to train but can be slow at inference time (driven by sampling results probabilistically multiple times, effectively increasing the computational burden linearly with the number of samples).


Explore the APIs
----------------

Explore the specific model implementations, architecture details, and code examples in their respective version documentation:

.. toctree::
    :maxdepth: 2

    Models-v1 <models>
    Models-v2 <models_v2>
