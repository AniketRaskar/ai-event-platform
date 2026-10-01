"""NeuMF learns both simple user–event matches and nonlinear preference patterns.

The GMF branch multiplies user and event factors element by element, while the
MLP branch learns deeper patterns from their concatenated representations. In
the hybrid version, event text and user interests provide side information;
randomly replacing event IDs with a learned unknown vector teaches the model
to recommend events it has never seen in positive training interactions.
"""
from __future__ import annotations
import torch
from torch import nn

class NeuMF(nn.Module):
    """Neural matrix factorization with optional event and user side features."""
    def __init__(self, n_users: int, n_events: int, event_content: torch.Tensor | None = None,
                 user_side_features: torch.Tensor | None = None, seen_event_mask: torch.Tensor | None = None,
                 latent_dim: int = 32, use_gmf: bool = True, use_mlp: bool = True,
                 use_content: bool = True, id_dropout: float = 0.3, use_user_side: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        if not use_gmf and not use_mlp:
            raise ValueError("At least one of use_gmf or use_mlp must be enabled")
        if not 0 <= id_dropout <= 1:
            raise ValueError("id_dropout must be between 0 and 1")
        self.n_users = n_users
        self.n_events = n_events
        self.latent_dim = latent_dim
        self.use_gmf = use_gmf
        self.use_mlp = use_mlp
        self.use_content = use_content
        self.id_dropout = id_dropout
        self.use_user_side = use_user_side and user_side_features is not None
        content_width = int(event_content.shape[1]) if event_content is not None else 384
        side_width = int(user_side_features.shape[1]) if self.use_user_side else 0
        self.user_embedding = nn.Embedding(n_users, latent_dim)
        self.event_embedding = nn.Embedding(n_events, latent_dim)
        self.unknown_event_embedding = nn.Parameter(torch.zeros(latent_dim))
        nn.init.normal_(self.user_embedding.weight, std=0.08)
        nn.init.normal_(self.event_embedding.weight, std=0.08)
        self.content_projection = nn.Linear(content_width, latent_dim) if use_content else None
        self.user_side_projection = nn.Linear(side_width, latent_dim) if self.use_user_side else None
        self.register_buffer("event_content", event_content.float() if event_content is not None else torch.zeros(n_events, content_width))
        self.register_buffer("user_side_features", user_side_features.float() if self.use_user_side else torch.zeros(n_users, 0))
        seen = seen_event_mask.bool() if seen_event_mask is not None else torch.ones(n_events, dtype=torch.bool)
        self.register_buffer("seen_event_mask", seen)
        user_width = latent_dim * (2 if self.use_user_side else 1)
        event_width = latent_dim * (2 if use_content else 1)
        self.mlp = nn.Sequential(nn.Linear(user_width + event_width, 64), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(32, 16), nn.ReLU()) if use_mlp else None
        fusion_width = (latent_dim if use_gmf else 0) + (16 if use_mlp else 0)
        self.output = nn.Linear(fusion_width, 1)

    def forward(self, user_indices: torch.Tensor, event_indices: torch.Tensor) -> torch.Tensor:
        """Return logits for aligned user and event indices."""
        user_id = self.user_embedding(user_indices)
        event_id = self.event_embedding(event_indices)
        replace_id = torch.zeros_like(event_indices, dtype=torch.bool)
        if self.training and self.id_dropout > 0:
            replace_id = torch.rand_like(event_indices, dtype=torch.float32) < self.id_dropout
        else:
            replace_id = ~self.seen_event_mask[event_indices]
        event_id = torch.where(replace_id.unsqueeze(-1), self.unknown_event_embedding.unsqueeze(0), event_id)
        user_parts = [user_id]
        event_parts = [event_id]
        user_side = None
        content = None
        if self.use_user_side:
            user_side = torch.relu(self.user_side_projection(self.user_side_features[user_indices]))
            user_parts.append(user_side)
        if self.use_content:
            content = torch.relu(self.content_projection(self.event_content[event_indices]))
            event_parts.append(content)
        outputs = []
        if self.use_gmf:
            # Pair matching factors directly; side information contributes its own
            # elementwise match when the corresponding counterpart is available.
            interactions = [user_id * event_id]
            if user_side is not None and content is not None:
                interactions.append(user_side * content)
            elif user_side is not None:
                interactions.append(user_side * event_id)
            elif content is not None:
                interactions.append(user_id * content)
            outputs.append(torch.stack(interactions).sum(dim=0))
        if self.use_mlp:
            outputs.append(self.mlp(torch.cat(user_parts + event_parts, dim=-1)))
        return self.output(torch.cat(outputs, dim=-1)).squeeze(-1)

    @torch.inference_mode()
    def score_pairs(self, user_indices: torch.Tensor, event_indices: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
        """Score aligned pairs in batches, forcing UNK IDs for unseen events."""
        was_training = self.training
        self.eval()
        scores = [self(u, e) for u, e in zip(user_indices.split(batch_size), event_indices.split(batch_size))]
        if was_training:
            self.train()
        return torch.cat(scores) if scores else torch.empty(0)
