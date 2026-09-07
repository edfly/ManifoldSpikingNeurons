"""
Sphere manifold S^{d-1} (radius r) operations.

Used by the v38 supplementary experiment (P2-7): extending the product
manifold H^{d_h} x R^{d_e} with a spherical component S^{d_s-1},
giving H^{d_h} x S^{d_s-1} x R^{d_e}.

Conventions follow msg_baseline.manifolds.PoincareManifold:
  - logmap(y, x) returns the tangent vector AT x pointing TOWARD y.
  - expmap(v, x) maps tangent vector v at x to the manifold.
  - origin(*size) returns a canonical base point.
"""
import torch


class SphereManifold:
    """Unit-radius sphere S^{d-1} embedded in R^d (radius fixed at 1.0)."""

    def __init__(self, dim: int, r: float = 1.0, eps: float = 1e-5):
        self.dim = dim
        self.r = r
        self.eps = eps

    # ── Basic ops ────────────────────────────────────────────────
    def origin(self, *size, device=None, **kwargs) -> torch.Tensor:
        """Canonical base point o = (r, 0, ..., 0)."""
        o = torch.zeros(*size, self.dim, device=device)
        o[..., 0] = self.r
        return o

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        """Radial projection back onto the sphere."""
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return self.r * x / norm

    def _tangent(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Project ambient vector v onto the tangent space at x."""
        # x is on the sphere: x . v component removed
        dot = (x * v).sum(dim=-1, keepdim=True)
        return v - (dot / (self.r ** 2)) * x

    def expmap(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Exponential map on the sphere.

        exp_x(v) = cos(||v||/r) x + sin(||v||/r) * (r/||v||) * v_tan
        """
        v = self._tangent(v, x)
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        y = torch.cos(v_norm / self.r) * x + \
            torch.sin(v_norm / self.r) * (self.r / v_norm) * v
        return self.projx(y)

    def logmap(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Logarithmic map: tangent vector at x pointing toward y."""
        # Geodesic distance
        dot = (x * y).sum(dim=-1, keepdim=True) / (self.r ** 2)
        dist = self.r * torch.acos(dot.clamp(-1.0 + self.eps, 1.0 - self.eps))
        # Tangent direction
        u = y - (dot / self.r ** 2) * x  # (x . y / r^2) scaling keeps u in tangent space
        u_norm = u.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return (dist / u_norm) * u

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Geodesic distance r * arccos(<x,y>/r^2)."""
        dot = (x * y).sum(dim=-1) / (self.r ** 2)
        return self.r * torch.acos(dot.clamp(-1.0 + self.eps, 1.0 - self.eps))

    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        """Map tangent vector at origin o=(r,0,..,0) to the sphere.

        v is (..., d-1) free tangent coordinates (the component along o
        is zero at the origin). exp_o(v) = (r cos(||v||/r),
        r sin(||v||/r) * v/||v||).
        """
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        y = torch.cat(
            [self.r * torch.cos(v_norm / self.r),
             self.r * torch.sin(v_norm / self.r) * v / v_norm],
            dim=-1)
        return self.projx(y)
