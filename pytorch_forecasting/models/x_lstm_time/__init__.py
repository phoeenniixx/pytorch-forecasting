from pytorch_forecasting.models.base_model import AutoRegressiveBaseModel
from copy import copy
from typing import Optional, Tuple, Union, Literal, Dict
from pytorch_forecasting.metrics import SMAPE, Metric
import torch
from torch import nn
from pytorch_forecasting.models.x_lstm_time.s_lstm.network import sLSTMNetwork
from pytorch_forecasting.models.x_lstm_time.m_lstm.network import mLSTMNetwork
from pytorch_forecasting.models.x_lstm_time.series_decomp import SeriesDecomposition


class xLSTMTime(AutoRegressiveBaseModel):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        xlstm_type: Literal["slstm", "mlstm"] = "slstm",
        num_layers: int = 1,
        decomposition_kernel: int = 25,
        input_projection_size: Optional[int] = None,
        dropout: float = 0.1,
        loss: Metric = SMAPE(),
        device: Optional[torch.device] = None,
        **kwargs,
    ):

        if "target" in kwargs:
            del kwargs["target"]
        if "target_lags" in kwargs:
            del kwargs["target_lags"]
        self.save_hyperparameters()
        super().__init__(loss=loss, **kwargs)

        if xlstm_type not in ["slstm", "mlstm"]:
            raise ValueError("xlstm_type must be either 'slstm' or 'mlstm'")

        self.xlstm_type = xlstm_type
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self._device)

        self.decomposition = SeriesDecomposition(decomposition_kernel)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        self.input_projection_size = input_projection_size or hidden_size
        self.input_linear = None

        if xlstm_type == "mlstm":
            self.lstm = mLSTMNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=hidden_size,
                dropout=dropout,
                device=self.device,
            )
        else:  # slstm
            self.lstm = sLSTMNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=hidden_size,
                dropout=dropout,
                device=self.device,
            )

        self.output_linear = nn.Linear(hidden_size, output_size)
        self.instance_norm = nn.InstanceNorm1d(output_size)

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        hidden_states: Optional[
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> Dict[str, torch.Tensor]:
        encoder_cont = x["encoder_cont"]
        batch_size, seq_len, n_features = encoder_cont.shape

        trend, seasonal = self.decomposition(encoder_cont)

        x = torch.cat([trend, seasonal], dim=-1)
        concatenated_features = x.shape[-1]

        if self.input_linear is None:
            self.input_linear = nn.Linear(concatenated_features, self.input_projection_size).to(self.device)

        x = self.input_linear(x)

        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)

        if hidden_states is None:
            hidden_states = self.lstm.init_hidden(batch_size)

        x = x.transpose(0, 1)
        output, hidden_states = self.lstm(x, *hidden_states)

        if isinstance(output, tuple):
            output = output[0]

        if output.dim() == 2:
            output = output.unsqueeze(0)

        output = self.output_linear(output)

        output = output.transpose(1, 2)
        output = self.instance_norm(output)
        output = output.transpose(1, 2)

        output = output[0, ..., : self.hparams.output_size]
        return self.to_network_output(prediction=output)

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        new_kwargs = copy(kwargs)
        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, SMAPE()))

        return super().from_dataset(dataset, **kwargs)