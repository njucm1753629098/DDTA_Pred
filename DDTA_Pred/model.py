import torch
import torch.nn as nn
import torch.nn.functional as F


class DiseaseTreatmentVectorModel(nn.Module):
    """Map a disease Word2Vec vector into the therapeutic herb vector space."""

    def __init__(self, vector_size=128, hidden_size=384, dropout=0.20):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(vector_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vector_size),
        )
        self.interaction_head = nn.Sequential(
            nn.Linear(vector_size * 4 + 1, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        self.temperature = nn.Parameter(torch.tensor(8.0))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, disease_vec, herb_vec):
        treatment_vec = self.generator(disease_vec)
        treatment_vec = F.normalize(treatment_vec, dim=1)
        disease_norm = F.normalize(disease_vec, dim=1)
        herb_norm = F.normalize(herb_vec, dim=1)
        cosine_score = torch.sum(treatment_vec * herb_norm, dim=1, keepdim=True)
        pair_features = torch.cat(
            [
                disease_norm,
                herb_norm,
                disease_norm * herb_norm,
                torch.abs(disease_norm - herb_norm),
                cosine_score,
            ],
            dim=1,
        )
        return cosine_score * self.temperature.clamp(1.0, 30.0) + self.interaction_head(pair_features) + self.bias

    def treatment_vector(self, disease_vec):
        with torch.no_grad():
            return F.normalize(self.generator(disease_vec), dim=1)
