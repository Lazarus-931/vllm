import torch
import pytest


def hadamard_ref(x: torch.Tensor) -> torch.Tensor:
    y = x.clone().float()
    n = y.shape[-1]
    h = 1
    while h < n:
        y = y.view(*y.shape[:-1], -1, 2, h)
        a = y[..., 0, :].clone()
        b = y[..., 1, :].clone()
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        y = y.view(*y.shape[:-3], n)
        h *= 2
    return y / (n ** 0.5)


def quantize_ref(x: torch.Tensor):
    boundaries = torch.tensor(
        [0.25822, 0.52241, 0.79955, 1.09929, 1.43714, 1.84354, 2.40081],
        dtype=x.dtype, device=x.device,
    )
    codebook = torch.tensor(
        [0.12821, 0.38806, 0.65619, 0.94222, 1.25616, 1.62429, 2.08578, 2.73276],
        dtype=x.dtype, device=x.device,
    )
    mag_idx = torch.bucketize(torch.abs(x), boundaries)
    sign = torch.where(x < 0, -1.0, 1.0).to(x.dtype)
    reconstructed = codebook[mag_idx] * sign
    sign_bit = (x < 0).to(torch.uint8)
    packed_idx = mag_idx.to(torch.uint8) | (sign_bit << 3)
    return packed_idx, reconstructed


def pack_ref(indices: torch.Tensor) -> torch.Tensor:
    lo = indices[..., 0::2]
    hi = indices[..., 1::2]
    return lo | (hi << 4)


def unpack_ref(packed: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    mag_lo = lo & 0x07
    mag_hi = hi & 0x07
    sign_lo = ((lo >> 3) & 1).float() * -2.0 + 1.0
    sign_hi = ((hi >> 3) & 1).float() * -2.0 + 1.0
    val_lo = codebook[mag_lo.long()] * sign_lo
    val_hi = codebook[mag_hi.long()] * sign_hi
    out = torch.stack([val_lo, val_hi], dim=-1)
    return out.view(*packed.shape[:-1], packed.shape[-1] * 2)


class TestHadamardRefSelfInverse:
    @pytest.mark.parametrize("head_size", [64, 128])
    def test_self_inverse(self, head_size):
        x = torch.randn(4, 8, head_size)
        h1 = hadamard_ref(x)
        h2 = hadamard_ref(h1)
        torch.testing.assert_close(h2, x.float(), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("head_size", [64, 128])
    def test_preserves_norm(self, head_size):
        x = torch.randn(4, 8, head_size)
        h = hadamard_ref(x)
        x_norms = torch.norm(x.float(), dim=-1)
        h_norms = torch.norm(h, dim=-1)
        torch.testing.assert_close(x_norms, h_norms, atol=1e-4, rtol=1e-4)


class TestQuantPackRoundtrip:
    def test_pack_unpack_identity(self):
        codebook = torch.tensor(
            [0.12821, 0.38806, 0.65619, 0.94222, 1.25616, 1.62429, 2.08578, 2.73276]
        )
        x = torch.randn(4, 8, 128)
        indices, _ = quantize_ref(x)
        packed = pack_ref(indices)
        unpacked = unpack_ref(packed, codebook)
        _, expected = quantize_ref(x)
        torch.testing.assert_close(unpacked, expected, atol=1e-5, rtol=1e-5)

    def test_quantization_error_bounded(self):
        x = torch.randn(1000, 8, 128)
        _, reconstructed = quantize_ref(x)
        mse = ((x - reconstructed) ** 2).mean()
        assert mse < 0.04, f"MSE {mse:.4f} too high for 4-bit Lloyd-Max on Gaussian"


class TestFullPipeline:
    @pytest.mark.parametrize("head_size", [64, 128])
    def test_hadamard_quant_dequant_inv_hadamard(self, head_size):
        codebook = torch.tensor(
            [0.12821, 0.38806, 0.65619, 0.94222, 1.25616, 1.62429, 2.08578, 2.73276]
        )
        x = torch.randn(16, 8, head_size)

        rotated = hadamard_ref(x)
        indices, _ = quantize_ref(rotated)
        packed = pack_ref(indices)
        unpacked = unpack_ref(packed, codebook)
        recovered = hadamard_ref(unpacked)

        mse = ((x.float() - recovered) ** 2).mean()
        cosine_sim = torch.nn.functional.cosine_similarity(
            x.float().view(-1, head_size),
            recovered.view(-1, head_size),
            dim=-1,
        ).mean()

        assert mse < 0.05, f"Full pipeline MSE {mse:.4f} too high"
        assert cosine_sim > 0.95, f"Cosine similarity {cosine_sim:.4f} too low"

    @pytest.mark.parametrize("head_size", [64, 128])
    def test_sign_preservation(self, head_size):
        x = torch.randn(8, 4, head_size)
        indices, recon = quantize_ref(x)
        signs_match = (x.sign() == recon.sign()) | (x == 0) | (recon == 0)
        pct = signs_match.float().mean()
        assert pct > 0.99, f"Sign mismatch rate too high: {1-pct:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
