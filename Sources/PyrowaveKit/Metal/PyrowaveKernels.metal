#include <metal_stdlib>
using namespace metal;

struct QuantizationConstants {
    uint count;
    float quantizationStep;
};

struct PlaneQuantizationDescriptor {
    uint originX;
    uint originY;
    uint validWidth;
    uint validHeight;
    uint stride;
    uint quantCode;
    float baseScale;
};

struct PlaneQuantizationConstants {
    uint descriptorCount;
};

struct SparseCoefficientEntry {
    uint destinationOffset;
    int coefficient;
    uint quantCode;
    uint qScaleCode;
};

struct SparseApplyConstants {
    uint entryCount;
    uint sampleCount;
};

struct DWTConstants {
    uint activeWidth;
    uint activeHeight;
    uint stride;
    uint phase;
};

constant float dwtAlpha = -1.586134342059924f;
constant float dwtBeta = -0.052980118572961f;
constant float dwtGamma = 0.882911075530934f;
constant float dwtDelta = 0.443506852043971f;
constant float dwtK = 1.230174104914001f;
constant float dwtInvK = 1.0f / 1.230174104914001f;

static inline int mirrorIndex(int index, int count) {
    if (count <= 1) {
        return 0;
    }
    if (index < 0) {
        return -index;
    }
    if (index >= count) {
        return 2 * count - index - 2;
    }
    return index;
}

kernel void pyrowave_quantize(
    device const float *input [[buffer(0)]],
    device short *output [[buffer(1)]],
    constant QuantizationConstants &constants [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= constants.count) {
        return;
    }

    float scaled = rint(input[index] / constants.quantizationStep);
    scaled = clamp(scaled, -32768.0f, 32767.0f);
    output[index] = short(scaled);
}

kernel void pyrowave_dequantize(
    device const short *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant QuantizationConstants &constants [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= constants.count) {
        return;
    }

    output[index] = float(input[index]) * constants.quantizationStep;
}

static inline uchar pyrowave_encode_8x8_scale(float scale) {
    float encoded = ceil((scale - 0.25f) * 8.0f);
    encoded = clamp(encoded, 0.0f, 15.0f);
    return uchar(encoded);
}

static inline uchar pyrowave_encode_8x8_scale_code(float maxScaledCoefficient) {
    if (maxScaledCoefficient < 1.0f) {
        return 6;
    }

    int exponent = int(floor(log2(maxScaledCoefficient - 0.25f))) + 1;
    float targetMax = exp2(float(exponent)) - 0.25f;
    return pyrowave_encode_8x8_scale(maxScaledCoefficient / targetMax);
}

static inline float pyrowave_decode_block_scale(uint quantCode) {
    int exponent = 4 - int(quantCode >> 3u);
    uint mantissa = quantCode & 7u;
    return (float(8u + mantissa) * exp2(float(20 + exponent))) / (8.0f * 1024.0f * 1024.0f);
}

static inline float pyrowave_decode_8x8_scale(uint code) {
    return float(code) / 8.0f + 0.25f;
}

kernel void pyrowave_quantize_plane_tiles(
    device const float *samples [[buffer(0)]],
    device short *coefficients [[buffer(1)]],
    device const PlaneQuantizationDescriptor *descriptors [[buffer(2)]],
    device uchar *qScaleCodes [[buffer(3)]],
    constant PlaneQuantizationConstants &constants [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    uint descriptorIndex = index >> 4u;
    uint smallBlock = index & 15u;
    if (descriptorIndex >= constants.descriptorCount) {
        return;
    }

    PlaneQuantizationDescriptor descriptor = descriptors[descriptorIndex];
    uint smallBlockX = smallBlock & 3u;
    uint smallBlockY = smallBlock >> 2u;
    uint localX0 = smallBlockX * 8u;
    uint localY0 = smallBlockY * 8u;
    uint validWidth = descriptor.validWidth > localX0 ? min(8u, descriptor.validWidth - localX0) : 0u;
    uint validHeight = descriptor.validHeight > localY0 ? min(8u, descriptor.validHeight - localY0) : 0u;
    if (validWidth == 0u || validHeight == 0u) {
        qScaleCodes[descriptorIndex * 16u + smallBlock] = 6;
        return;
    }

    uint originX = descriptor.originX + localX0;
    uint originY = descriptor.originY + localY0;
    float maxScaledCoefficient = 0.0f;
    for (uint y = 0; y < validHeight; ++y) {
        uint row = (originY + y) * descriptor.stride + originX;
        for (uint x = 0; x < validWidth; ++x) {
            maxScaledCoefficient = max(maxScaledCoefficient, fabs(samples[row + x] * descriptor.baseScale));
        }
    }

    uchar qScaleCode = pyrowave_encode_8x8_scale_code(maxScaledCoefficient);
    qScaleCodes[descriptorIndex * 16u + smallBlock] = qScaleCode;
    float decoded8x8Scale = float(qScaleCode) / 8.0f + 0.25f;
    float quantScale = 1.0f / decoded8x8Scale;

    for (uint y = 0; y < validHeight; ++y) {
        uint row = (originY + y) * descriptor.stride + originX;
        for (uint x = 0; x < validWidth; ++x) {
            float scaled = trunc(samples[row + x] * descriptor.baseScale * quantScale);
            scaled = clamp(scaled, -32768.0f, 32767.0f);
            coefficients[row + x] = short(scaled);
        }
    }
}

kernel void pyrowave_apply_sparse_coefficients(
    device float *samples [[buffer(0)]],
    device const SparseCoefficientEntry *entries [[buffer(1)]],
    constant SparseApplyConstants &constants [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= constants.entryCount) {
        return;
    }

    SparseCoefficientEntry entry = entries[index];
    if (entry.destinationOffset >= constants.sampleCount) {
        return;
    }

    float value = float(entry.coefficient);
    if (value > 0.0f) {
        value += 0.5f;
    } else if (value < 0.0f) {
        value -= 0.5f;
    }
    samples[entry.destinationOffset] = value
        * pyrowave_decode_block_scale(entry.quantCode)
        * pyrowave_decode_8x8_scale(entry.qScaleCode);
}

kernel void pyrowave_dwt_lift_rows(
    device float *samples [[buffer(0)]],
    constant DWTConstants &constants [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.activeWidth || y >= constants.activeHeight) {
        return;
    }

    bool odd = (x & 1u) != 0u;
    bool active = false;
    float factor = 0.0f;
    switch (constants.phase) {
    case 0: active = odd; factor = dwtAlpha; break;
    case 1: active = !odd; factor = dwtBeta; break;
    case 2: active = odd; factor = dwtGamma; break;
    case 3: active = !odd; factor = dwtDelta; break;
    default: break;
    }

    uint index = y * constants.stride + x;
    if (constants.phase < 4) {
        if (!active) {
            return;
        }
        int left = mirrorIndex(int(x) - 1, int(constants.activeWidth));
        int right = mirrorIndex(int(x) + 1, int(constants.activeWidth));
        samples[index] += factor * (samples[y * constants.stride + uint(left)] + samples[y * constants.stride + uint(right)]);
    } else {
        samples[index] *= odd ? dwtK : dwtInvK;
    }
}

kernel void pyrowave_dwt_lift_columns(
    device float *samples [[buffer(0)]],
    constant DWTConstants &constants [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.activeWidth || y >= constants.activeHeight) {
        return;
    }

    bool odd = (y & 1u) != 0u;
    bool active = false;
    float factor = 0.0f;
    switch (constants.phase) {
    case 0: active = odd; factor = dwtAlpha; break;
    case 1: active = !odd; factor = dwtBeta; break;
    case 2: active = odd; factor = dwtGamma; break;
    case 3: active = !odd; factor = dwtDelta; break;
    default: break;
    }

    uint index = y * constants.stride + x;
    if (constants.phase < 4) {
        if (!active) {
            return;
        }
        int top = mirrorIndex(int(y) - 1, int(constants.activeHeight));
        int bottom = mirrorIndex(int(y) + 1, int(constants.activeHeight));
        samples[index] += factor * (samples[uint(top) * constants.stride + x] + samples[uint(bottom) * constants.stride + x]);
    } else {
        samples[index] *= odd ? dwtK : dwtInvK;
    }
}

kernel void pyrowave_dwt_pack_rows(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant DWTConstants &constants [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.activeWidth || y >= constants.activeHeight) {
        return;
    }

    uint lowCount = (constants.activeWidth + 1u) >> 1u;
    uint packedX = ((x & 1u) == 0u) ? (x >> 1u) : (lowCount + (x >> 1u));
    output[y * constants.stride + packedX] = input[y * constants.stride + x];
}

kernel void pyrowave_dwt_pack_columns(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant DWTConstants &constants [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.activeWidth || y >= constants.activeHeight) {
        return;
    }

    uint lowCount = (constants.activeHeight + 1u) >> 1u;
    uint packedY = ((y & 1u) == 0u) ? (y >> 1u) : (lowCount + (y >> 1u));
    output[packedY * constants.stride + x] = input[y * constants.stride + x];
}

kernel void pyrowave_dwt_unpack_rows(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant DWTConstants &constants [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.activeWidth || y >= constants.activeHeight) {
        return;
    }

    uint lowCount = (constants.activeWidth + 1u) >> 1u;
    uint packedX = ((x & 1u) == 0u) ? (x >> 1u) : (lowCount + (x >> 1u));
    output[y * constants.stride + x] = input[y * constants.stride + packedX];
}

kernel void pyrowave_dwt_unpack_columns(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant DWTConstants &constants [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.activeWidth || y >= constants.activeHeight) {
        return;
    }

    uint lowCount = (constants.activeHeight + 1u) >> 1u;
    uint packedY = ((y & 1u) == 0u) ? (y >> 1u) : (lowCount + (y >> 1u));
    output[y * constants.stride + x] = input[packedY * constants.stride + x];
}

kernel void pyrowave_idwt_lift_rows(
    device float *samples [[buffer(0)]],
    constant DWTConstants &constants [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.activeWidth || y >= constants.activeHeight) {
        return;
    }

    bool odd = (x & 1u) != 0u;
    uint index = y * constants.stride + x;
    if (constants.phase == 0) {
        samples[index] *= odd ? dwtInvK : dwtK;
        return;
    }

    bool active = false;
    float factor = 0.0f;
    switch (constants.phase) {
    case 1: active = !odd; factor = dwtDelta; break;
    case 2: active = odd; factor = dwtGamma; break;
    case 3: active = !odd; factor = dwtBeta; break;
    case 4: active = odd; factor = dwtAlpha; break;
    default: break;
    }

    if (!active) {
        return;
    }
    int left = mirrorIndex(int(x) - 1, int(constants.activeWidth));
    int right = mirrorIndex(int(x) + 1, int(constants.activeWidth));
    samples[index] -= factor * (samples[y * constants.stride + uint(left)] + samples[y * constants.stride + uint(right)]);
}

kernel void pyrowave_idwt_lift_columns(
    device float *samples [[buffer(0)]],
    constant DWTConstants &constants [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint x = position.x;
    uint y = position.y;
    if (x >= constants.activeWidth || y >= constants.activeHeight) {
        return;
    }

    bool odd = (y & 1u) != 0u;
    uint index = y * constants.stride + x;
    if (constants.phase == 0) {
        samples[index] *= odd ? dwtInvK : dwtK;
        return;
    }

    bool active = false;
    float factor = 0.0f;
    switch (constants.phase) {
    case 1: active = !odd; factor = dwtDelta; break;
    case 2: active = odd; factor = dwtGamma; break;
    case 3: active = !odd; factor = dwtBeta; break;
    case 4: active = odd; factor = dwtAlpha; break;
    default: break;
    }

    if (!active) {
        return;
    }
    int top = mirrorIndex(int(y) - 1, int(constants.activeHeight));
    int bottom = mirrorIndex(int(y) + 1, int(constants.activeHeight));
    samples[index] -= factor * (samples[uint(top) * constants.stride + x] + samples[uint(bottom) * constants.stride + x]);
}
