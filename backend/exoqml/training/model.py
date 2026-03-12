from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as nnf

try:
    import pennylane as qml

    HAS_PENNYLANE = True
except Exception:
    qml = None
    HAS_PENNYLANE = False


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = nnf.gelu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = nnf.gelu(x + residual)
        return x


class TransitResNet1D(nn.Module):
    def __init__(self, base_channels: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv1d(1, c, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(c),
            nn.GELU(),
        )
        self.layer1 = nn.Sequential(
            ResidualBlock1D(c, c, stride=1, dropout=dropout),
            ResidualBlock1D(c, c, stride=1, dropout=dropout),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock1D(c, c * 2, stride=2, dropout=dropout),
            ResidualBlock1D(c * 2, c * 2, stride=1, dropout=dropout),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock1D(c * 2, c * 4, stride=2, dropout=dropout),
            ResidualBlock1D(c * 4, c * 4, stride=1, dropout=dropout),
        )
        self.layer4 = nn.Sequential(
            ResidualBlock1D(c * 4, c * 4, stride=2, dropout=dropout),
            ResidualBlock1D(c * 4, c * 4, stride=1, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 4, c * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.head(x)


class ViewEncoder1D(nn.Module):
    def __init__(self, base_channels: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv1d(1, c, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(c),
            nn.GELU(),
            ResidualBlock1D(c, c, stride=1, dropout=dropout),
            ResidualBlock1D(c, c * 2, stride=2, dropout=dropout),
            ResidualBlock1D(c * 2, c * 2, stride=1, dropout=dropout),
            ResidualBlock1D(c * 2, c * 4, stride=2, dropout=dropout),
            ResidualBlock1D(c * 4, c * 4, stride=1, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.out_dim = c * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransitMultiViewNet(nn.Module):
    def __init__(self, scalar_dim: int = 4, base_channels: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.global_encoder = ViewEncoder1D(base_channels=base_channels, dropout=dropout)
        self.local_encoder = ViewEncoder1D(base_channels=base_channels, dropout=dropout)
        hidden = base_channels * 4
        self.scalar_head = nn.Sequential(
            nn.Linear(scalar_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        fusion_dim = (self.global_encoder.out_dim * 2) + hidden
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode_features(
        self,
        global_view: torch.Tensor,
        local_view: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> torch.Tensor:
        global_feat = self.global_encoder(global_view)
        local_feat = self.local_encoder(local_view)
        scalar_feat = self.scalar_head(scalar_features)
        return torch.cat([global_feat, local_feat, scalar_feat], dim=1)

    def forward(self, global_view: torch.Tensor, local_view: torch.Tensor, scalar_features: torch.Tensor) -> torch.Tensor:
        fused = self.encode_features(global_view, local_view, scalar_features)
        return self.head(fused)


class QuantumCircuitHead(nn.Module):
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, device_name: str = "default.qubit") -> None:
        super().__init__()
        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane is not available in this environment.")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device_name = device_name
        self.dev = qml.device(device_name, wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class TransitHybridQMLNet(nn.Module):
    def __init__(
        self,
        scalar_dim: int = 4,
        base_channels: int = 32,
        dropout: float = 0.2,
        n_qubits: int = 4,
        n_q_layers: int = 2,
        q_device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.global_encoder = ViewEncoder1D(base_channels=base_channels, dropout=dropout)
        self.local_encoder = ViewEncoder1D(base_channels=base_channels, dropout=dropout)
        hidden = base_channels * 4
        self.scalar_head = nn.Sequential(
            nn.Linear(scalar_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        fusion_dim = (self.global_encoder.out_dim * 2) + hidden
        self.pre_qml = nn.Sequential(
            nn.Linear(fusion_dim, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, n_qubits),
            nn.Tanh(),
        )
        self.q_scale = math.pi / 2.0
        self.quantum_head = QuantumCircuitHead(n_qubits=n_qubits, n_layers=n_q_layers, device_name=q_device)
        self.post_qml = nn.Sequential(
            nn.Linear(n_qubits, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.n_qubits = n_qubits
        self.n_q_layers = n_q_layers
        self.q_device = q_device

    def encode_features(
        self,
        global_view: torch.Tensor,
        local_view: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> torch.Tensor:
        global_feat = self.global_encoder(global_view)
        local_feat = self.local_encoder(local_view)
        scalar_feat = self.scalar_head(scalar_features)
        return torch.cat([global_feat, local_feat, scalar_feat], dim=1)

    def forward(self, global_view: torch.Tensor, local_view: torch.Tensor, scalar_features: torch.Tensor) -> torch.Tensor:
        fused = self.encode_features(global_view, local_view, scalar_features)
        q_inputs = self.pre_qml(fused) * self.q_scale
        q_outputs = self.quantum_head(q_inputs)
        return self.post_qml(q_outputs)


class TransitResidualQMLNet(nn.Module):
    def __init__(
        self,
        scalar_dim: int = 4,
        base_channels: int = 32,
        dropout: float = 0.2,
        n_qubits: int = 4,
        n_q_layers: int = 2,
        q_device: str = "default.qubit",
        residual_alpha_init: float = 0.15,
    ) -> None:
        super().__init__()
        self.global_encoder = ViewEncoder1D(base_channels=base_channels, dropout=dropout)
        self.local_encoder = ViewEncoder1D(base_channels=base_channels, dropout=dropout)
        hidden = base_channels * 4
        self.scalar_head = nn.Sequential(
            nn.Linear(scalar_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        fusion_dim = (self.global_encoder.out_dim * 2) + hidden
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.pre_qml = nn.Sequential(
            nn.Linear(fusion_dim + 1, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, n_qubits),
            nn.Tanh(),
        )
        self.q_scale = math.pi / 2.0
        self.quantum_head = QuantumCircuitHead(n_qubits=n_qubits, n_layers=n_q_layers, device_name=q_device)
        self.delta_head = nn.Sequential(
            nn.Linear(n_qubits, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        residual_alpha_init = float(min(max(residual_alpha_init, 1e-3), 1.0 - 1e-3))
        alpha_logit = math.log(residual_alpha_init / (1.0 - residual_alpha_init))
        self.residual_gate_logit = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))
        self.n_qubits = n_qubits
        self.n_q_layers = n_q_layers
        self.q_device = q_device

    def encode_features(
        self,
        global_view: torch.Tensor,
        local_view: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> torch.Tensor:
        global_feat = self.global_encoder(global_view)
        local_feat = self.local_encoder(local_view)
        scalar_feat = self.scalar_head(scalar_features)
        return torch.cat([global_feat, local_feat, scalar_feat], dim=1)

    def forward_with_components(
        self,
        global_view: torch.Tensor,
        local_view: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fused = self.encode_features(global_view, local_view, scalar_features)
        classical_logit = self.head(fused)
        q_inputs = self.pre_qml(torch.cat([fused, classical_logit.detach()], dim=1)) * self.q_scale
        q_outputs = self.quantum_head(q_inputs)
        residual_logit = self.delta_head(q_outputs)
        residual_alpha = torch.sigmoid(self.residual_gate_logit)
        total_logit = classical_logit + (residual_alpha * residual_logit)
        return total_logit, classical_logit, residual_logit, residual_alpha

    def forward(self, global_view: torch.Tensor, local_view: torch.Tensor, scalar_features: torch.Tensor) -> torch.Tensor:
        total_logit, _, _, _ = self.forward_with_components(global_view, local_view, scalar_features)
        return total_logit
