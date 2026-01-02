import torch
import triton
import triton.language as tl

# ============================================================================
# SwiGLU: silu(gate) * up in one pass
# ============================================================================

@triton.jit
def _swiglu_fwd(a_ptr, b_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    a_ptr += row * stride
    b_ptr += row * stride
    c_ptr += row * stride

    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    a = tl.load(a_ptr + offs, mask=mask, other=0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0)

    # silu(a) * b
    silu_a = a * tl.sigmoid(a)
    c = silu_a.to(b.dtype) * b
    tl.store(c_ptr + offs, c, mask=mask)


@triton.jit
def _swiglu_bwd(dc_ptr, a_ptr, b_ptr, stride, n_cols: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    dc_ptr += row * stride
    a_ptr += row * stride
    b_ptr += row * stride

    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    dc = tl.load(dc_ptr + offs, mask=mask, other=0)
    a = tl.load(a_ptr + offs, mask=mask, other=0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0)

    sig_a = tl.sigmoid(a)
    silu_a = a * sig_a
    db = dc * silu_a
    da = dc * (silu_a * (1 - sig_a) + sig_a) * b

    tl.store(a_ptr + offs, da, mask=mask)
    tl.store(b_ptr + offs, db, mask=mask)


def _get_block_warps(n_cols):
    block = triton.next_power_of_2(n_cols)
    block = min(max(block, 32), 65536)
    warps = 4 if block <= 1024 else (8 if block <= 4096 else 16)
    return block, warps


class SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        gate, up = gate.contiguous(), up.contiguous()
        shape = gate.shape
        n_cols = shape[-1]
        gate_flat = gate.view(-1, n_cols)
        up_flat = up.view(-1, n_cols)
        out = torch.empty_like(gate_flat)

        BLOCK, warps = _get_block_warps(n_cols)
        _swiglu_fwd[(gate_flat.shape[0],)](
            gate_flat, up_flat, out, out.stride(0),
            n_cols=n_cols, BLOCK=BLOCK, num_warps=warps
        )
        ctx.save_for_backward(gate_flat, up_flat)
        ctx.shape = shape
        return out.view(*shape)

    @staticmethod
    def backward(ctx, dc):
        gate, up = ctx.saved_tensors
        dc = dc.contiguous().view_as(gate)
        n_cols = gate.shape[-1]

        BLOCK, warps = _get_block_warps(n_cols)
        _swiglu_bwd[(gate.shape[0],)](
            dc, gate, up, dc.stride(0),
            n_cols=n_cols, BLOCK=BLOCK, num_warps=warps
        )
        return gate.view(*ctx.shape), up.view(*ctx.shape)


def swiglu(gate, up):
    return SwiGLU.apply(gate, up)


# ============================================================================
# Fused Add + RMSNorm: x = rmsnorm(x + residual)
# ============================================================================

@triton.jit
def _fused_add_rms_fwd(
    out_ptr, resid_out_ptr, x_ptr, resid_ptr, w_ptr, rstd_ptr,
    stride, n_cols, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    out_ptr += row * stride
    resid_out_ptr += row * stride
    x_ptr += row * stride
    resid_ptr += row * stride

    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(x_ptr + offs, mask=mask, other=0)
    r = tl.load(resid_ptr + offs, mask=mask, other=0)
    w = tl.load(w_ptr + offs, mask=mask, other=0)

    # fused add
    s = x + r
    tl.store(resid_out_ptr + offs, s, mask=mask)

    # rmsnorm
    s_f32 = s.to(tl.float32)
    var = tl.sum(s_f32 * s_f32, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(rstd_ptr + row, rstd)

    out = (s_f32 * rstd).to(s.dtype) * w
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _fused_add_rms_bwd(
    dx_ptr, dr_ptr, dout_ptr, resid_ptr, w_ptr, rstd_ptr,
    stride, n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    dx_ptr += row * stride
    dr_ptr += row * stride
    dout_ptr += row * stride
    resid_ptr += row * stride

    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    dout = tl.load(dout_ptr + offs, mask=mask, other=0).to(tl.float32)
    s = tl.load(resid_ptr + offs, mask=mask, other=0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)

    # d(rmsnorm)
    ds = dout * w * rstd
    c = tl.sum(ds * s, axis=0) / n_cols * rstd * rstd
    ds = ds - s * c

    tl.store(dx_ptr + offs, ds.to(dx_ptr.dtype.element_ty), mask=mask)
    tl.store(dr_ptr + offs, ds.to(dr_ptr.dtype.element_ty), mask=mask)


class FusedAddRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight, eps=1e-6):
        x, residual = x.contiguous(), residual.contiguous()
        shape = x.shape
        n_cols = shape[-1]
        x_flat = x.view(-1, n_cols)
        r_flat = residual.view(-1, n_cols)
        n_rows = x_flat.shape[0]

        out = torch.empty_like(x_flat)
        resid_out = torch.empty_like(x_flat)
        rstd = torch.empty(n_rows, device=x.device, dtype=torch.float32)

        BLOCK, warps = _get_block_warps(n_cols)
        _fused_add_rms_fwd[(n_rows,)](
            out, resid_out, x_flat, r_flat, weight, rstd,
            x_flat.stride(0), n_cols, eps,
            BLOCK=BLOCK, num_warps=warps
        )
        ctx.save_for_backward(resid_out, weight, rstd)
        ctx.shape = shape
        return out.view(*shape), resid_out.view(*shape)

    @staticmethod
    def backward(ctx, dout, dresid_out):
        resid, weight, rstd = ctx.saved_tensors
        dout = (dout + dresid_out).contiguous()
        n_cols = resid.shape[-1]
        dout_flat = dout.view(-1, n_cols)

        dx = torch.empty_like(dout_flat)
        dr = torch.empty_like(dout_flat)

        BLOCK, warps = _get_block_warps(n_cols)
        _fused_add_rms_bwd[(dout_flat.shape[0],)](
            dx, dr, dout_flat, resid, weight, rstd,
            dout_flat.stride(0), n_cols,
            BLOCK=BLOCK, num_warps=warps
        )
        return dx.view(*ctx.shape), dr.view(*ctx.shape), None, None


def fused_add_rms_norm(x, residual, weight, eps=1e-6):
    return FusedAddRMSNorm.apply(x, residual, weight, eps)


# ============================================================================
# Fused LayerNorm: for vision encoder
# ============================================================================

@triton.jit
def _layernorm_fwd(
    out_ptr, x_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr,
    stride, n_cols, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    out_ptr += row * stride
    x_ptr += row * stride

    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    var = tl.sum((x - mean) * (x - mean), axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    out = (x - mean) * rstd * w + b
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _layernorm_bwd(
    dx_ptr, dout_ptr, x_ptr, w_ptr, mean_ptr, rstd_ptr,
    stride, n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    dx_ptr += row * stride
    dout_ptr += row * stride
    x_ptr += row * stride

    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    dout = tl.load(dout_ptr + offs, mask=mask, other=0).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0).to(tl.float32)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)

    xhat = (x - mean) * rstd
    wdy = w * dout
    c1 = tl.sum(wdy, axis=0) / n_cols
    c2 = tl.sum(wdy * xhat, axis=0) / n_cols
    dx = rstd * (wdy - c1 - xhat * c2)

    tl.store(dx_ptr + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-6):
        x = x.contiguous()
        shape = x.shape
        n_cols = shape[-1]
        x_flat = x.view(-1, n_cols)
        n_rows = x_flat.shape[0]

        out = torch.empty_like(x_flat)
        mean = torch.empty(n_rows, device=x.device, dtype=torch.float32)
        rstd = torch.empty(n_rows, device=x.device, dtype=torch.float32)

        BLOCK, warps = _get_block_warps(n_cols)
        _layernorm_fwd[(n_rows,)](
            out, x_flat, weight, bias, mean, rstd,
            x_flat.stride(0), n_cols, eps,
            BLOCK=BLOCK, num_warps=warps
        )
        ctx.save_for_backward(x_flat, weight, mean, rstd)
        ctx.shape = shape
        return out.view(*shape)

    @staticmethod
    def backward(ctx, dout):
        x, weight, mean, rstd = ctx.saved_tensors
        dout = dout.contiguous()
        n_cols = x.shape[-1]
        dout_flat = dout.view(-1, n_cols)

        dx = torch.empty_like(x)

        BLOCK, warps = _get_block_warps(n_cols)
        _layernorm_bwd[(x.shape[0],)](
            dx, dout_flat, x, weight, mean, rstd,
            x.stride(0), n_cols,
            BLOCK=BLOCK, num_warps=warps
        )
        # dw, db computed via reduction over rows
        dw = (dout_flat * ((x - mean.unsqueeze(1)) * rstd.unsqueeze(1))).sum(0)
        db = dout_flat.sum(0)
        return dx.view(*ctx.shape), dw, db, None


def layer_norm(x, weight, bias, eps=1e-6):
    return LayerNorm.apply(x, weight, bias, eps)


